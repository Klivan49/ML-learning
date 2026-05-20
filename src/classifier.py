import os
import joblib
import sys
from typing import Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from filename_parser import FileNameParser

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/tfidf_logreg.joblib")

_content_classifier = None


def _get_content_classifier():
    global _content_classifier
    if _content_classifier is None:
        try:
            from content import ContentClassifier
            _content_classifier = ContentClassifier()
        except Exception as e:
            print(f"Warning: Content classifier not loaded: {e}", file=sys.stderr)
    return _content_classifier


class FileNameClassifier:
    CONFIDENCE_THRESHOLD = 0.7

    def __init__(self, model_path=MODEL_PATH, use_content: bool = True):
        self.model_path = model_path
        self.model = None
        self.vectorizer = None
        self._load_model()

        self.use_content = use_content
        self.content_classifier = None
        if use_content:
            self.content_classifier = _get_content_classifier()

    def _load_model(self):
        obj = joblib.load(self.model_path)
        self.vectorizer = obj["vectorizer"]
        self.model = obj["clf"]

    def predict_by_filename(self, filename: str) -> Tuple[str, float]:
        if self.model:
            tok = " ".join(FileNameParser.tokenize(filename))
            X_vec = self.vectorizer.transform([tok])
            pred = self.model.predict(X_vec)[0]

            try:
                proba = self.model.predict_proba(X_vec)[0]
                confidence = float(max(proba))
            except Exception:
                confidence = 0.7

            return pred, confidence

        return self._rule_based_filename_predict(filename)

    def _rule_based_filename_predict(self, filename: str) -> Tuple[str, float]:
        tokens = FileNameParser.tokenize(filename)
        
        study_keywords = {
            "лабораторная", "лаб", "лр", "пз", "лекция", "семинар", "кр",
            "курсовая", "реферат", "контрольная", "тест", "экзамен",
            "практика", "задание", "вариант", "титульный", "отчёт",
            "laboratory", "lecture", "seminar", "assignment", "report",
            "essay", "exam", "coursework", "study",
        }
        
        for token in tokens:
            if token in study_keywords:
                return "Study", 0.8
        
        return "NotStudy", 0.5

    def predict_by_content(self, filepath: str) -> Tuple[Optional[str], float, str]:
        if not self.content_classifier:
            return None, 0.0, "not_available"

        return self.content_classifier.classify(filepath, use_ml=True)

    def verify_extension(self, filepath: str) -> Tuple[bool, Optional[str]]:
        if not self.content_classifier:
            return True, None

        return self.content_classifier.verify_extension(filepath)

    def predict(
        self, filename: str, filepath: str = None, use_content: bool = None
    ) -> str:
        if use_content is None:
            use_content = self.use_content

        filename_pred, filename_conf = self.predict_by_filename(filename)

        if use_content and filepath and os.path.exists(filepath):
            content_pred, content_conf, method = self.predict_by_content(filepath)

            if content_pred and content_conf >= self.CONFIDENCE_THRESHOLD:
                return self._combine_predictions(
                    filename_pred, filename_conf,
                    content_pred, content_conf, method,
                    filepath
                )

            if content_pred and method == "ml":
                return self._combine_predictions(
                    filename_pred, filename_conf,
                    content_pred, content_conf, method,
                    filepath
                )

        return self._finalize_prediction(filename_pred, filename_conf, filepath)

    def _combine_predictions(
        self,
        filename_pred: str,
        filename_conf: float,
        content_pred: str,
        content_conf: float,
        method: str,
        filepath: str
    ) -> str:
        if filename_pred == "Study" and content_pred in {"Study", "NotStudy"}:
            if filename_conf > content_conf:
                return self._map_to_category(filename_pred, filepath)
            return self._map_to_category(content_pred, filepath)

        if content_conf > filename_conf:
            return self._map_to_category(content_pred, filepath)

        return self._map_to_category(filename_pred, filepath)

    def _finalize_prediction(self, pred: str, confidence: float, filepath: str) -> str:
        if filepath and os.path.exists(filepath) and self.content_classifier:
            ext = os.path.splitext(filepath)[1].lower()
            content_type, _, _ = self.content_classifier.classify(filepath, use_ml=False)

            if content_type and content_type != "Unknown":
                return self._map_to_category(pred, filepath, content_type)

        return self._map_to_category(pred, filepath)

    def _map_to_category(
        self, pred: str, filepath: str = None, content_type: str = None
    ) -> str:
        if pred == "Study":
            return "Study"

        if filepath:
            ext = os.path.splitext(filepath)[1].lower()
        else:
            ext = ""

        if ext in {".ppt", ".pptx", ".key", ".odp"}:
            return "Presentations"
        if ext in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}:
            return "Images"
        if ext in {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2"}:
            return "Archives"
        if ext in {".mp3", ".wav", ".flac", ".mp4", ".avi", ".mov", ".mkv"}:
            return "Media"
        if ext in {".torrent"}:
            return "Others"
        if ext in {".sav", ".spv", ".erwin", ".mdb"}:
            return "Others"
        if ext in {".doc", ".docx", ".pdf", ".txt", ".xls", ".xlsx", ".csv"}:
            if content_type in {"Invoice", "Report"}:
                return content_type
            return "Documents"

        if content_type == "TextDocument":
            return "Documents"
        if content_type == "OfficeDocument":
            return "Documents"
        if content_type == "PDFDocument":
            return "Documents"

        return "Others"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/classifier.py [--no-content] ")
        exit(1)

    use_content = "--no-content" not in sys.argv
    fname = [arg for arg in sys.argv[1:] if not arg.startswith("--")][0]
    
    clf = FileNameClassifier(use_content=use_content)

    if os.path.exists(fname):
        category = clf.predict(os.path.basename(fname), filepath=fname)
    else:
        category = clf.predict(fname)

    print(f"{fname} -> {category}")