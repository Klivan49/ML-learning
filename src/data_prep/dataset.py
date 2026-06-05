import os
import random
import string
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")
from configs.config import CONFIG
from src.features.features import extract_all_features, FEATURE_COLUMNS


def collect_real_files(root_dir: str) -> List[str]:
    file_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            full_path = os.path.join(dirpath, fname)
            if os.path.isfile(full_path):
                file_paths.append(full_path)
    return file_paths


def assign_class_from_path(file_path: str, root_dir: str) -> str:
    rel_path = os.path.relpath(file_path, root_dir)
    parts = rel_path.split(os.sep)
    if len(parts) > 1:
        folder = parts[0].lower()
        for cat in CONFIG.target_columns:
            if folder == cat or folder.startswith(cat):
                return cat
    return "other"


def build_real_dataset(root_dir: str, output_csv: Optional[str] = None) -> pd.DataFrame:
    file_paths = collect_real_files(root_dir)
    records = []
    for fp in tqdm(file_paths, desc="Extracting features from real files"):
        try:
            feats = extract_all_features(fp)
            target = assign_class_from_path(fp, root_dir)
            feats["file_path"] = fp
            feats["file_name"] = os.path.basename(fp)
            feats["extension"] = os.path.splitext(fp)[1].lstrip(".")
            feats["target_class"] = target
            records.append(feats)
        except Exception:
            continue

    df = pd.DataFrame(records)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df


def generate_synthetic_text_content(category: str, num_words: int = 50) -> str:
    corpus = CONFIG.synthetic_corpora
    words = corpus.get(category) or corpus.get("general") or corpus.get("other") or ["data", "text"]
    return " ".join(random.choice(words) for _ in range(num_words))


def generate_synthetic_binary_content(category: str, size: int) -> bytes:
    if category == "images":
        return bytes(random.randint(0, 255) for _ in range(min(size, 4096)))
    elif category == "archives":
        data = bytes(random.randint(0, 255) for _ in range(min(size, 4096)))
        return bytes([0x50, 0x4B, 0x03, 0x04]) + data[:4092]
    elif category in ("media", "mp3", "wav", "flac", "mp4", "avi", "mkv"):
        return bytes([0xFF, 0xFB]) + bytes(random.randint(0, 255) for _ in range(min(size - 2, 4094)))
    else:
        return bytes(random.randint(0, 255) for _ in range(min(size, 4096)))


def _generate_synthetic_record(
    category: str,
    idx: int,
    synthetic_dir: str,
) -> Dict:
    name_template = random.choice(CONFIG.synthetic_filename_patterns.get(category, ["file_{n}"]))
    date = f"{random.randint(2019, 2025):04d}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
    name = random.choice(string.ascii_lowercase) + str(random.randint(1, 999))
    topics_general = ["sales", "marketing", "tech", "hr", "finance", "product"]
    topics_edu = ["algebra", "physics", "prog", "chem", "history", "english"]
    topic = random.choice(topics_edu if CONFIG.profile == "education" else topics_general)
    year = str(random.randint(2019, 2025))
    n = str(random.randint(1, 9999))

    file_name = name_template.format(date=date, name=name, topic=topic, year=year, n=n)
    ext_list = CONFIG.extension_to_category.get(category, ["txt"])
    ext = random.choice(ext_list)
    full_name = f"{file_name}.{ext}"
    size = random.randint(*CONFIG.size_ranges.get(category, (100, 1_000_000)))

    os.makedirs(synthetic_dir, exist_ok=True)
    file_path = os.path.join(synthetic_dir, full_name)

    if ext in CONFIG.text_extensions:
        content = generate_synthetic_text_content(category)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        content = generate_synthetic_binary_content(category, size)
        with open(file_path, "wb") as f:
            f.write(content)

    feats = extract_all_features(file_path)
    feats["file_path"] = file_path
    feats["file_name"] = full_name
    feats["extension"] = ext
    feats["target_class"] = category
    return feats


def build_synthetic_dataset(
    num_samples: int,
    output_csv: str,
    synthetic_dir: str = "synthetic_data",
    seed: int = 42,
) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    categories = CONFIG.target_columns
    samples_per_cat = max(1, num_samples // len(categories))
    records = []

    for cat in categories:
        for i in tqdm(range(samples_per_cat), desc=f"Generating {cat}"):
            rec = _generate_synthetic_record(cat, i, os.path.join(synthetic_dir, cat))
            records.append(rec)

    remaining = num_samples - len(records)
    for i in tqdm(range(remaining), desc="Generating extra samples"):
        cat = random.choice(categories)
        rec = _generate_synthetic_record(cat, i, os.path.join(synthetic_dir, cat))
        records.append(rec)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df
