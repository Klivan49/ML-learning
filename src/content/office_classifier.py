import os
import re
import zipfile
import joblib
from typing import Tuple, Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import logging

from . import BaseContentClassifier

logger = logging.getLogger(__name__)


class OfficeClassifier(BaseContentClassifier):
    CATEGORIES = [
        "Study",
        "NotStudy",
        "Invoice",
        "Report",
        "Presentation",
        "Spreadsheet",
    ]

    MODEL_SUBDIR = "office"

    TEXT_CONTENT_PATHS = {
        ".docx": ["word/document.xml", "word/steps/*.xml"],
        ".xlsx": ["xl/sharedStrings.xml", "xl/worksheets/*.xml"],
        ".pptx": ["ppt/slides/*.xml", "ppt/notesSlides/*.xml"],
    }

    def __init__(self, model_dir: str = None):
        subdir = os.path.join(model_dir or os.path.join(
            os.path.dirname(__file__), "../../models/content"
        ), self.MODEL_SUBDIR)
        super().__init__(subdir)

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

    def extract_text_from_office(self, filepath: str) -> str:
        ext = os.path.splitext(filepath)[1].lower()
        text_parts = []

        try:
            with zipfile.ZipFile(filepath, "r") as zf:
                for name in zf.namelist():
                    if name.endswith(".xml"):
                        try:
                            content = zf.read(name).decode("utf-8", errors="ignore")
                            content = re.sub(r"<[^>]+>", " ", content)
                            content = re.sub(r"\s+", " ", content)
                            content = content.strip()
                            if content:
                                text_parts.append(content)
                        except Exception:
                            continue
        except Exception as e:
            logger.warning(f"Failed to extract from {filepath}: {e}")
            return ""

        full_text = " ".join(text_parts[:50])
        return full_text[:100000]

    def extract_metadata(self, filepath: str) -> dict:
        metadata = {
            "has_author": False,
            "has_title": False,
            "has_keywords": False,
            "has_comments": False,
            "slide_count": 0,
            "sheet_count": 0,
            "has_table": False,
            "has_image": False,
        }

        try:
            with zipfile.ZipFile(filepath, "r") as zf:
                files = zf.namelist()

                if "docProps/core.xml" in files:
                    try:
                        core = zf.read("docProps/core.xml").decode("utf-8", errors="ignore")
                        metadata["has_author"] = bool(re.search(r"<dc:creator>", core))
                        metadata["has_title"] = bool(re.search(r"<dc:title>", core))
                        metadata["has_keywords"] = bool(re.search(r"<cp:keywords>", core))
                        metadata["has_comments"] = bool(re.search(r"<dc:description>", core))
                    except Exception:
                        pass

                if "ppt/slides" in str(files):
                    slide_files = [f for f in files if f.startswith("ppt/slides/slide")]
                    metadata["slide_count"] = len(slide_files)

                if "xl/worksheets" in str(files):
                    sheet_files = [f for f in files if f.startswith("xl/worksheets/sheet")]
                    metadata["sheet_count"] = len(sheet_files)

                metadata["has_table"] = any("table" in f for f in files)
                metadata["has_image"] = any(
                    "media" in f or "image" in f for f in files
                )

        except Exception as e:
            logger.warning(f"Failed to extract metadata from {filepath}: {e}")

        return metadata

    def extract_features(self, text: str, metadata: dict) -> dict:
        features = {}

        study_patterns = [
            r"\b(задание|лабораторная|лекция|семинар|пз|лр|кр|курсовая|реферат|экзамен|тест|контрольная)\b",
            r"\b(study|laboratory|lecture|seminar|assignment|report|essay|exam|coursework)\b",
            r"\bвариант\d+",
            r"\b\d+ПЗ\d*|\b\d+ЛР\d*",
            r"титульный",
            r"отчёт",
        ]
        features["has_study_keywords"] = any(
            re.search(p, text, re.IGNORECASE) for p in study_patterns
        )

        invoice_patterns = [
            r"\b(инвойс|invoice|счёт|акт|квитанция|оплата|payment)\b",
            r"\$\d+|руб\.?\d+",
            r"сумма|итого|total",
        ]
        features["has_invoice_keywords"] = any(
            re.search(p, text, re.IGNORECASE) for p in invoice_patterns
        )

        report_patterns = [
            r"\b(отчёт|report|анализ|статистика|summary)\b",
            r"\d{4}[-/]\d{2}[-/]\d{2}",
        ]
        features["has_report_keywords"] = any(
            re.search(p, text, re.IGNORECASE) for p in report_patterns
        )

        features["slide_count"] = metadata.get("slide_count", 0)
        features["sheet_count"] = metadata.get("sheet_count", 0)
        features["has_table"] = metadata.get("has_table", False)
        features["has_image"] = metadata.get("has_image", False)
        features["has_author"] = metadata.get("has_author", False)

        return features

    def classify(
        self, filepath: str, file_size: int = None
    ) -> Tuple[Optional[str], float, str]:
        if file_size is None:
            file_size = os.path.getsize(filepath)

        text = self.extract_text_from_office(filepath)
        if not text:
            return None, 0.0, "empty_file"

        metadata = self.extract_metadata(filepath)
        features = self.extract_features(text, metadata)

        if self.model and self.vectorizer:
            try:
                X_vec = self.vectorizer.transform([text])
                prediction = self.model.predict(X_vec)[0]
                proba = self.model.predict_proba(X_vec)[0]
                confidence = float(max(proba))

                return prediction, confidence, "ml"
            except Exception:
                pass

        return self._rule_based_classify(features, file_size, filepath)

    def _rule_based_classify(
        self, features: dict, file_size: int, filepath: str
    ) -> Tuple[Optional[str], float, str]:
        ext = os.path.splitext(filepath)[1].lower()

        if ext in {".pptx", ".odp"}:
            if features.get("slide_count", 0) > 0:
                return "Presentation", 0.7, "extension"

        if ext in {".xlsx", ".ods"}:
            if features.get("sheet_count", 0) > 0:
                return "Spreadsheet", 0.7, "extension"

        if features.get("has_invoice_keywords"):
            return "Invoice", 0.7, "rules"
        if features.get("has_study_keywords"):
            return "Study", 0.7, "rules"
        if features.get("has_report_keywords"):
            return "Report", 0.6, "rules"
        if features.get("has_author") and not features.get("has_study_keywords"):
            return "NotStudy", 0.5, "rules"

        size_category = self._get_file_size_category(file_size)
        if ext in {".docx", ".odt"}:
            if size_category in ["tiny", "small"]:
                return "NotStudy", 0.4, "size"

        return "NotStudy", 0.5, "default"


def train_office_classifier(data_dir: str, output_dir: str):
    texts = []
    labels = []

    office_extensions = {".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp"}

    for filename in os.listdir(data_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in office_extensions:
            continue

        filepath = os.path.join(data_dir, filename)
        if os.path.isfile(filepath):
            clf = OfficeClassifier()
            text = clf.extract_text_from_office(filepath)
            if text:
                texts.append(text[:50000])
                
                if "study" in filename.lower() or "учеб" in filename.lower():
                    label = "Study"
                elif "invoice" in filename.lower() or "счёт" in filename.lower():
                    label = "Invoice"
                elif "report" in filename.lower() or "отчёт" in filename.lower():
                    label = "Report"
                elif ".pptx" in ext:
                    label = "Presentation"
                elif ".xlsx" in ext:
                    label = "Spreadsheet"
                else:
                    label = "NotStudy"
                    
                labels.append(label)

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