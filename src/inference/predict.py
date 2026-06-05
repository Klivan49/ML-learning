import os
import joblib
import pandas as pd
from typing import Dict, List, Optional, Tuple

from src.features.features import extract_all_features, get_feature_columns, detect_content_type
from configs.config import CONFIG

EDU_FOLDER_MAP = {
    "лабораторные": "Лабораторные",
    "практические": "Практические",
    "методички": "Методички",
    "курсовые": "Курсовые",
    "код": "Код",
}


class FileClassifier:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def predict_file(self, file_path: str) -> Tuple[str, Dict[str, float]]:
        feats = extract_all_features(file_path)
        feature_vector = self._feats_to_df(feats)
        pred = self.model.predict(feature_vector)[0]
        probs = self.model.predict_proba(feature_vector)[0]
        prob_dict = dict(zip(self.model.classes_, probs))
        return pred, prob_dict

    def predict_file_verbose(self, file_path: str) -> Tuple[str, Dict[str, float], str, str, bool]:
        feats = extract_all_features(file_path)
        feature_vector = self._feats_to_df(feats)
        pred = self.model.predict(feature_vector)[0]
        probs = self.model.predict_proba(feature_vector)[0]
        prob_dict = dict(zip(self.model.classes_, probs))
        detected_type, detected_cat = detect_content_type(file_path)
        if CONFIG.profile == "education":
            doc_classes = {"лабораторные", "практические", "методички", "курсовые"}
            if detected_cat == "code":
                suspicious = (pred != "код")
            elif detected_cat == "documents":
                suspicious = (pred not in doc_classes)
            elif detected_cat == "other":
                suspicious = False
            else:
                suspicious = True
        else:
            suspicious = (pred != detected_cat and detected_cat != "other")
        return pred, prob_dict, detected_type, detected_cat, suspicious

    def predict_features(self, feats_dict: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
        feature_vector = self._feats_to_df(feats_dict)
        pred = self.model.predict(feature_vector)[0]
        probs = self.model.predict_proba(feature_vector)[0]
        prob_dict = dict(zip(self.model.classes_, probs))
        return pred, prob_dict

    def _feats_to_df(self, feats: Dict[str, float]) -> pd.DataFrame:
        row = {col: feats.get(col, 0.0) for col in get_feature_columns()}
        return pd.DataFrame([row])

    @staticmethod
    def move_file(file_path: str, target_dir: str, dry_run: bool = False) -> str:
        if dry_run:
            return f"[ТЕСТ] Будет перемещён {file_path} -> {target_dir}"
        os.makedirs(target_dir, exist_ok=True)
        dest = os.path.join(target_dir, os.path.basename(file_path))
        if os.path.exists(dest):
            stem, ext = os.path.splitext(dest)
            counter = 1
            while os.path.exists(f"{stem}_{counter}{ext}"):
                counter += 1
            dest = f"{stem}_{counter}{ext}"
        os.rename(file_path, dest)
        return f"Перемещён: {file_path} -> {dest}"

    def sort_file(self, file_path: str, output_root: str, dry_run: bool = False,
                  flag_suspicious: bool = True) -> str:
        if not os.path.isfile(file_path):
            return f"Error: {file_path} is not a file"
        pred, probs, det_type, det_cat, suspicious = self.predict_file_verbose(file_path)
        if suspicious:
            msg = (f"⚠ ПОДОЗРИТЕЛЬНЫЙ: {os.path.basename(file_path)} → предсказан '{pred}' "
                   f"но содержимое — {det_type} ({det_cat})")
            if flag_suspicious:
                target_dir = os.path.join(output_root, "_suspicious")
                result = self.move_file(file_path, target_dir, dry_run)
                return f"{msg}\n  → перемещён в _suspicious/\n  {result}"
            return msg
        target_dir = os.path.join(output_root, pred)
        return self.move_file(file_path, target_dir, dry_run)

    def sort_directory(
        self, dir_path: str, output_root: str,
        recursive: bool = True, dry_run: bool = False,
        min_size: int = 0, max_size: int = 0,
        allowed_extensions: Optional[List[str]] = None,
        flag_suspicious: bool = True,
    ) -> List[str]:
        results = []
        file_iter = self._walk_files(dir_path, recursive)
        for fp in file_iter:
            if min_size > 0 or max_size > 0:
                try:
                    sz = os.path.getsize(fp)
                    if min_size > 0 and sz < min_size:
                        continue
                    if max_size > 0 and sz > max_size:
                        continue
                except OSError:
                    continue
            if allowed_extensions:
                ext = os.path.splitext(fp)[1].lstrip(".").lower()
                if ext not in allowed_extensions:
                    continue
            result = self.sort_file(fp, output_root, dry_run, flag_suspicious)
            results.append(result)
        return results

    @staticmethod
    def _walk_files(dir_path: str, recursive: bool):
        if recursive:
            for dirpath, _, filenames in os.walk(dir_path):
                for fname in filenames:
                    yield os.path.join(dirpath, fname)
        else:
            for fname in os.listdir(dir_path):
                fp = os.path.join(dir_path, fname)
                if os.path.isfile(fp):
                    yield fp
