import os
import re
import joblib
import numpy as np
from typing import Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import BaseContentClassifier


class TextClassifier(BaseContentClassifier):
    CATEGORIES = [
        "Study",
        "NotStudy",
        "Code",
        "Config",
        "Log",
        "Data",
    ]

    MODEL_SUBDIR = "text"

    def __init__(self, model_dir: str = None):
        subdir = os.path.join(model_dir or os.path.join(
            os.path.dirname(__file__), "../../models/content"
        ), self.MODEL_SUBDIR)
        super().__init__(subdir)

    def preprocess_text(self, text: str) -> str:
        text = text[:50000]
        lines = text.split("\n")
        processed_lines = []
        for line in lines[:100]:
            line = re.sub(r"\s+", " ", line.strip())
            if line and len(line) > 2:
                processed_lines.append(line)
        return " ".join(processed_lines)

    def extract_features(self, text: str) -> dict:
        features = {}
        features["char_count"] = len(text)
        features["line_count"] = text.count("\n") + 1
        features["word_count"] = len(text.split())
        features["avg_word_len"] = (
            features["char_count"] / features["word_count"]
            if features["word_count"] > 0 else 0
        )

        code_patterns = [
            r"^\s*(def|class|function|const|let|var|import|export|public|private)\s",
            r"^\s*#include",
            r"^\s*package\s+",
            r"^\s*using\s+namespace",
            r"\{\s*$",
            r"^\s*//|^\s*#",
        ]
        features["has_code"] = any(re.search(p, text, re.MULTILINE) for p in code_patterns)

        study_patterns = [
            r"\b(задание|лабораторная|лекция|семинар|пз|лр|кр|курсовая|реферат|экзамен|тест|контрольная)\b",
            r"\b(study|laboratory|lecture|seminar|assignment|report|essay|exam)\b",
            r"\bвариант\d+",
            r"\b\d+ПЗ\d*|\b\d+ЛР\d*",
        ]
        features["has_study_keywords"] = any(
            re.search(p, text, re.IGNORECASE) for p in study_patterns
        )

        config_patterns = [
            r"^\s*(server|database|host|port|api|endpoint|config)\s*=",
            r"^\s*\{\s*\"",
            r"^\s*\[",
        ]
        features["has_config"] = any(
            re.search(p, text, re.IGNORECASE | re.MULTILINE) for p in config_patterns
        )

        log_patterns = [
            r"^\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}",
            r"\[(ERROR|WARN|INFO|DEBUG)\]",
            r"\b(error|warning|exception|fail)\b",
        ]
        features["has_log"] = any(
            re.search(p, text, re.IGNORECASE) for p in log_patterns
        )

        return features

    def _load_model(self):
        model_path = os.path.join(self.model_dir, "model.joblib")
        vec_path = os.path.join(self.model_dir, "vectorizer.joblib")

        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
            except Exception:
                self.model = None

        if os.path.exists(vec_path):
            try:
                self.vectorizer = joblib.load(vec_path)
            except Exception:
                self.vectorizer = None

    def classify(
        self, filepath: str, file_size: int = None
    ) -> Tuple[Optional[str], float, str]:
        if file_size is None:
            file_size = os.path.getsize(filepath)

        text = self.extract_text(filepath)
        if not text:
            return None, 0.0, "empty_file"

        processed_text = self.preprocess_text(text)
        features = self.extract_features(text)

        if self.model and self.vectorizer:
            try:
                X_vec = self.vectorizer.transform([processed_text])
                prediction = self.model.predict(X_vec)[0]
                proba = self.model.predict_proba(X_vec)[0]
                confidence = float(max(proba))

                return prediction, confidence, "ml"
            except Exception:
                pass

        return self._rule_based_classify(features, file_size)

    def _rule_based_classify(
        self, features: dict, file_size: int
    ) -> Tuple[Optional[str], float, str]:
        if features.get("has_code"):
            return "Code", 0.7, "rules"
        if features.get("has_study_keywords"):
            return "Study", 0.6, "rules"
        if features.get("has_log"):
            return "Log", 0.7, "rules"
        if features.get("has_config"):
            return "Config", 0.6, "rules"

        size_category = self._get_file_size_category(file_size)
        if size_category in ["tiny", "small"]:
            return "Data", 0.4, "size"

        return "NotStudy", 0.5, "default"


def train_text_classifier(data_dir: str, output_dir: str):
    texts = []
    labels = []

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if os.path.isfile(filepath):
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                texts.append(text[:50000])
                label = "Study"
                if "code" in filename.lower():
                    label = "Code"
                elif "config" in filename.lower():
                    label = "Config"
                elif "log" in filename.lower() or "data" in filename.lower():
                    label = "Data"
                else:
                    label = "NotStudy"
                labels.append(label)
            except Exception:
                continue

    if not texts:
        return False

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(X, labels)

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(output_dir, "model.joblib"))
    joblib.dump(vectorizer, os.path.join(output_dir, "vectorizer.joblib"))

    return True