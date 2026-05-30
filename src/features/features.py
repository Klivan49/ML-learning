import os
import re
import math
import hashlib
import numpy as np
from typing import Dict, List, Optional
from collections import Counter
from configs.config import CONFIG


def extract_filename_features(file_name: str) -> Dict[str, float]:
    name_stem, ext = os.path.splitext(file_name)
    name_stem = name_stem.lower()
    ext = ext.lower().lstrip(".")

    tokens = re.findall(r"[a-zA-Z]+|\d+", name_stem)
    char_count = len(name_stem)
    token_count = len(tokens)
    digit_count = sum(c.isdigit() for c in name_stem)
    has_date = bool(re.search(r"\d{4}[-_]\d{2}[-_]\d{2}|\d{2}[-_]\d{2}[-_]\d{4}", name_stem))
    word_count = sum(1 for t in tokens if t.isalpha())

    kw_features = {}
    for cat, keywords in CONFIG.category_keywords.items():
        kw_features[f"kw_{cat}"] = 1.0 if any(kw in name_stem for kw in keywords) else 0.0

    return {
        "filename_len": char_count,
        "filename_token_count": token_count,
        "filename_digit_ratio": digit_count / max(char_count, 1),
        "filename_word_count": word_count,
        "filename_has_date": 1.0 if has_date else 0.0,
        **kw_features,
    }


def extract_size_features(size_bytes: int) -> Dict[str, float]:
    log_size = math.log1p(size_bytes)
    kb = size_bytes / 1024
    mb = size_bytes / (1024 * 1024)

    if size_bytes < 1024:
        size_class = "tiny"
    elif size_bytes < 1024 * 100:
        size_class = "small"
    elif size_bytes < 1024 * 1024 * 10:
        size_class = "medium"
    elif size_bytes < 1024 * 1024 * 100:
        size_class = "large"
    else:
        size_class = "huge"

    return {
        "size_bytes": size_bytes,
        "log_size": log_size,
        "size_kb": kb,
        "size_mb": mb,
        "size_tiny": 1.0 if size_class == "tiny" else 0.0,
        "size_small": 1.0 if size_class == "small" else 0.0,
        "size_medium": 1.0 if size_class == "medium" else 0.0,
        "size_large": 1.0 if size_class == "large" else 0.0,
        "size_huge": 1.0 if size_class == "huge" else 0.0,
    }


def extract_extension_features(extension: str) -> Dict[str, float]:
    ext = extension.lower().lstrip(".") if extension else ""
    ext_group = CONFIG.extension_group_map.get(ext, "other")

    cat_features = {}
    for cat, exts in CONFIG.extension_to_category.items():
        cat_features[f"ext_cat_{cat}"] = 1.0 if ext in exts else 0.0

    return {
        "ext_group_code": 1.0 if ext_group == "code" else 0.0,
        "ext_group_doc": 1.0 if ext_group == "doc" else 0.0,
        "ext_group_img": 1.0 if ext_group == "img" else 0.0,
        "ext_group_arc": 1.0 if ext_group == "arc" else 0.0,
        "ext_group_audio": 1.0 if ext_group == "audio" else 0.0,
        "ext_group_video": 1.0 if ext_group == "video" else 0.0,
        "ext_group_other": 1.0 if ext_group == "other" else 0.0,
        **cat_features,
    }


def compute_byte_entropy(content: bytes) -> float:
    if not content:
        return 0.0
    counts = np.bincount(np.frombuffer(content, dtype=np.uint8), minlength=256)
    probs = counts / counts.sum()
    ent = -np.sum(p * np.log2(p) for p in probs if p > 0)
    return float(ent)


OFFICE_EXTENSIONS = {
    "docx", "docm", "dotx", "dotm",
    "xlsx", "xlsm", "xltx", "xltm",
    "pptx", "pptm", "potx", "potm", "ppsx", "ppsm",
}

MAGIC_SIGNATURES: list[tuple[bytes, str, str]] = [
    (b"\x89PNG\r\n\x1a\n",        "png", "images"),
    (b"\xff\xd8\xff",             "jpg", "images"),
    (b"GIF87a",                   "gif", "images"),
    (b"GIF89a",                   "gif", "images"),
    (b"BM",                       "bmp", "images"),
    (b"%PDF",                     "pdf", "documents"),
    (b"PK\x03\x04",               "zip", "archives"),
    (b"PK\x05\x06",               "zip", "archives"),
    (b"Rar!\x1a\x07",             "rar", "archives"),
    (b"\x1f\x8b",                 "gz",  "archives"),
    (b"BZh",                      "bz2", "archives"),
    (b"7z\xbc\xaf'\x1c",          "7z",  "archives"),
    (b"\x1f\x9d",                 "z",   "archives"),
    (b"\xff\xfb",                 "mp3", "media"),
    (b"\xff\xf3",                 "mp3", "media"),
    (b"\xff\xf2",                 "mp3", "media"),
    (b"ID3",                      "mp3", "media"),
    (b"RIFF",                     "wav", "media"),
    (b"\x00\x00\x00\x18ftyp",     "mp4", "media"),
    (b"\x00\x00\x00\x1cftyp",     "mp4", "media"),
    (b"OggS",                     "ogg", "media"),
    (b"FLAC",                     "flac", "media"),
    (b"\x1aE\xdf\xa3",            "mkv", "media"),
    (b"MO\x00\x00",               "avi", "media"),
    (b"\x7fELF",                  "elf", "code"),
    (b"#!",                       "script", "code"),
    (b"<!DOCTYPE html",           "html", "code"),
    (b"<html",                    "html", "code"),
    (b"<?xml",                    "xml",  "code"),
    (b"{\n",                      "json", "code"),
    (b"",                         "unknown", "other"),
]


def detect_content_type(file_path: str) -> tuple[str, str]:
    try:
        with open(file_path, "rb") as f:
            header = f.read(20)
    except OSError:
        return "unknown", "other"

    if not header:
        return "empty", "other"

    ext = os.path.splitext(file_path)[1].lstrip(".").lower()
    code_exts = {'py', 'js', 'ts', 'java', 'c', 'cpp', 'cs', 'go', 'rs', 'rb', 'php', 'swift', 'kt', 'scala', 'sh', 'bash', 'sql', 'html', 'css', 'scss', 'less', 'r', 'm', 'pl', 'lua', 'hs'}

    for sig, type_name, category in MAGIC_SIGNATURES:
        if sig and header.startswith(sig):
            if sig.startswith(b"PK") and ext in OFFICE_EXTENSIONS:
                return "ooxml", "documents"
            return type_name, category

    printable = sum(32 <= b < 127 for b in header) / len(header)
    zero = header.count(0) / len(header)
    if printable > 0.8 and zero < 0.05:
        if ext in code_exts:
            return "script", "code"
        return "text", "documents"
    return "binary", "other"


def extract_content_features(
    file_path: str,
    extension: str,
    max_text_bytes: int = 100_000,
) -> Dict[str, float]:
    ext = extension.lower().lstrip(".")
    detected_type, detected_cat = detect_content_type(file_path)

    result = {
        "ext_mismatch": 1.0 if ext not in CONFIG.extension_to_category.get(detected_cat, []) and detected_cat != "other" else 0.0,
        f"detected_cat_{detected_cat}": 1.0,
        "detected_is_text": 1.0 if detected_type in ("text", "script", "json", "xml", "html") or detected_cat == "code" else 0.0,
    }
    for cat in DETECTION_CATEGORIES:
        if f"detected_cat_{cat}" not in result:
            result[f"detected_cat_{cat}"] = 0.0

    if detected_cat == "documents" or detected_type in ("text", "script", "html", "xml", "json"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read(max_text_bytes)
            words = text.split()
            lines = text.splitlines()
            unique_words = len(set(words))
            result["text_length"] = len(text)
            result["text_word_count"] = len(words)
            result["text_line_count"] = len(lines)
            result["text_unique_word_ratio"] = unique_words / max(len(words), 1)
            result["text_avg_word_len"] = np.mean([len(w) for w in words]) if words else 0.0
            result["text_has_content"] = 1.0
        except Exception:
            result.update({"text_length": 0, "text_word_count": 0, "text_line_count": 0,
                           "text_unique_word_ratio": 0, "text_avg_word_len": 0, "text_has_content": 0.0})
    else:
        result.update({"text_length": 0, "text_word_count": 0, "text_line_count": 0,
                       "text_unique_word_ratio": 0, "text_avg_word_len": 0, "text_has_content": 0.0})

    try:
        with open(file_path, "rb") as f:
            raw = f.read(4096)
        entropy = compute_byte_entropy(raw)
        first_bytes_hash = int(hashlib.md5(raw[:256]).hexdigest()[:8], 16) % 1000
        zero_byte_ratio = raw.count(0) / max(len(raw), 1)
        printable_ratio = sum(32 <= b < 127 for b in raw) / max(len(raw), 1)

        result["binary_entropy"] = entropy
        result["binary_first_bytes_hash"] = first_bytes_hash / 1000.0
        result["binary_zero_ratio"] = zero_byte_ratio
        result["binary_printable_ratio"] = printable_ratio
        result["binary_has_content"] = 1.0
    except Exception:
        result.update({"binary_entropy": 0, "binary_first_bytes_hash": 0,
                       "binary_zero_ratio": 0, "binary_printable_ratio": 0, "binary_has_content": 0.0})

    return result


EDUCATION_KEYWORDS = {
    "лабораторные": ["лабораторная", "лабораторны", "лаб", "labwork", "ход_работы", "ход работы", "цель_работы", "цель работы", "отчёт", "отчет", "измерение"],
    "практические": ["практическая", "практическ", "пз_", "задание", "вариант", "решение", "задача", "упражнен", "семинар"],
    "методички":    ["методичка", "методическ", "лекция", "лекц", "тема", "теорема", "определение", "пособи", "guide", "manual"],
    "курсовые":     ["курсовая", "курсовой", "курсач", "coursework", "диплом", "реферат", "введение", "глава_", "заключение"],
    "код":          ["def ", "import ", "class ", "return ", "if __", "for ", "while ", "print(", "function", "algorithm", "алгоритм"],
}

def extract_text_keywords(file_path: str, target_columns: list, max_bytes: int = 20000) -> Dict[str, float]:
    text = ""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(max_bytes)
    except Exception:
        pass

    text_lower = text.lower()
    words = text_lower.split()
    word_count = max(len(words), 1)

    result = {}
    for cat in target_columns:
        kws = EDUCATION_KEYWORDS.get(cat, [])
        count = sum(text_lower.count(kw) for kw in kws)
        result[f"text_kw_{cat}"] = min(count / word_count * 100, 1.0)

    return result


def extract_all_features(file_path: str) -> Dict[str, float]:
    file_name = os.path.basename(file_path)
    _, ext = os.path.splitext(file_name)
    ext = ext.lstrip(".")

    try:
        size_bytes = os.path.getsize(file_path)
    except OSError:
        size_bytes = 0

    feats = {}
    feats.update(extract_filename_features(file_name))
    feats.update(extract_size_features(size_bytes))
    feats.update(extract_extension_features(ext))
    feats.update(extract_content_features(file_path, ext))
    if CONFIG.profile == "education":
        feats.update(extract_text_keywords(file_path, CONFIG.target_columns))
    return feats


DETECTION_CATEGORIES = ["documents", "images", "archives", "code", "media", "other"]


def get_feature_columns():
    return (
        ["filename_len", "filename_token_count", "filename_digit_ratio",
         "filename_word_count", "filename_has_date"]
        + [f"kw_{cat}" for cat in CONFIG.target_columns]
        + ["size_bytes", "log_size", "size_kb", "size_mb",
           "size_tiny", "size_small", "size_medium", "size_large", "size_huge",
           "ext_group_code", "ext_group_doc", "ext_group_img",
           "ext_group_arc", "ext_group_audio", "ext_group_video", "ext_group_other"]
        + [f"ext_cat_{cat}" for cat in CONFIG.target_columns]
        + ["ext_mismatch", "detected_is_text"]
        + [f"detected_cat_{cat}" for cat in DETECTION_CATEGORIES]
        + [f"text_kw_{cat}" for cat in CONFIG.target_columns]
        + ["text_length", "text_word_count", "text_line_count",
           "text_unique_word_ratio", "text_avg_word_len", "text_has_content",
           "binary_entropy", "binary_first_bytes_hash",
           "binary_zero_ratio", "binary_printable_ratio", "binary_has_content"]
    )


FEATURE_COLUMNS = get_feature_columns()
