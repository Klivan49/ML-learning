import os
import re
import joblib
from typing import Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import logging

from . import BaseContentClassifier

logger = logging.getLogger(__name__)


def extract_text_from_pdf(filepath: str) -> str:
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(filepath)
        return text[:100000]
    except ImportError:
        logger.warning("pdfminer.six not installed, trying fallback")
    except Exception as e:
        logger.warning(f"Failed to extract text from PDF {filepath}: {e}")

    try:
        import PyPDF2
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text_parts = []
            for page in reader.pages[:20]:
                try:
                    text_parts.append(page.extract_text())
                except Exception:
                    continue
            return " ".join(text_parts)
    except Exception:
        pass

    return ""


class PdfClassifier(BaseContentClassifier):
    CATEGORIES = [
        "Study",
        "NotStudy",
        "Invoice",
        "Report",
        "Book",
        "Manual",
    ]

    MODEL_SUBDIR = "pdf"

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

    def extract_text(self, filepath: str) -> str:
        text = extract_text_from_pdf(filepath)
        if not text:
            try:
                with open(filepath, "rb") as f:
                    header = f.read(100)
                    if b"%PDF" not in header:
                        return ""
            except Exception:
                pass
        return text

    def extract_features(self, text: str) -> dict:
        features = {}

        study_patterns = [
            r"\b(задание|лабораторная|лекция|семинар|пз|лр|кр|курсовая|реферат|экзамен|тест|контрольная)\b",
            r"\b(study|laboratory|lecture|seminar|assignment|report|essay|exam|coursework)\b",
            r"\bвариант\d+",
            r"\b\d+ПЗ\d*|\b\d+ЛР\d*",
            r"титульный лист",
            r"отчёт по (лабораторной|практике|курсовой)",
            r"условие задачи",
        ]
        features["has_study_keywords"] = any(
            re.search(p, text, re.IGNORECASE) for p in study_patterns
        )

        invoice_patterns = [
            r"\b(инвойс|invoice|счёт|акт|квитанция|оплата|payment|receipt)\b",
            r"\$\d+|руб\.?\d+",
            r"сумма|итого|total|amount",
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

        book_patterns = [
            r"\b(глава|chapter|раздел|section)\s+\d+",
            r"\bоглавление|содержание|contents\b",
            r"\b\d+\s+[а-яА-Я]+\s+\d{4}",
            r"издательство",
        ]
        features["is_book"] = any(
            re.search(p, text, re.IGNORECASE) for p in book_patterns
        )

        manual_patterns = [
            r"\b(руководство|manual|инструкция|guide|how to|справка|help)\b",
            r"version \d+\.\d+",
            r"\bcopyright\b",
        ]
        features["is_manual"] = any(
            re.search(p, text, re.IGNORECASE) for p in manual_patterns
        )

        features["page_count"] = text.count("\f") + 1

        return features

    def classify(
        self, filepath: str, file_size: int = None
    ) -> Tuple[Optional[str], float, str]:
        if file_size is None:
            file_size = os.path.getsize(filepath)

        text = self.extract_text(filepath)
        if not text:
            return None, 0.0, "empty_file"

        features = self.extract_features(text)

        if self.model and self.vectorizer:
            try:
                X_vec = self.vectorizer.transform([text])
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
        if features.get("has_invoice_keywords"):
            return "Invoice", 0.7, "rules"
        if features.get("has_study_keywords"):
            return "Study", 0.7, "rules"
        if features.get("is_book"):
            return "Book", 0.6, "rules"
        if features.get("is_manual"):
            return "Manual", 0.6, "rules"
        if features.get("has_report_keywords"):
            return "Report", 0.6, "rules"

        size_category = self._get_file_size_category(file_size)
        if size_category in ["huge", "massive"]:
            return "Book", 0.5, "size"
        elif size_category in ["tiny", "small"]:
            return "NotStudy", 0.4, "size"

        return "NotStudy", 0.5, "default"


def train_pdf_classifier(data_dir: str, output_dir: str):
    texts = []
    labels = []

    for filename in os.listdir(data_dir):
        if not filename.lower().endswith(".pdf"):
            continue

        filepath = os.path.join(data_dir, filename)
        if os.path.isfile(filepath):
            clf = PdfClassifier()
            text = clf.extract_text(filepath)
            if text:
                texts.append(text[:50000])
                
                if "study" in filename.lower() or "учеб" in filename.lower():
                    label = "Study"
                elif "invoice" in filename.lower() or "счёт" in filename.lower():
                    label = "Invoice"
                elif "report" in filename.lower() or "отчёт" in filename.lower():
                    label = "Report"
                elif "book" in filename.lower() or "книга" in filename.lower():
                    label = "Book"
                elif "manual" in filename.lower() or "руководство" in filename.lower():
                    label = "Manual"
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