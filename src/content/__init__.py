import os
import re
from typing import List, Optional, Tuple, Dict, Any
import logging
import joblib

logger = logging.getLogger(__name__)


class ContentClassificationResult:
    def __init__(self, category: str, confidence: float, method: str):
        self.category = category
        self.confidence = confidence
        self.method = method
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "confidence": self.confidence,
            "method": self.method,
        }


class BaseContentClassifier:
    MODEL_FILENAME = "model.joblib"
    VECTORIZER_FILENAME = "vectorizer.joblib"

    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or os.path.join(
            os.path.dirname(__file__), "../../models/content"
        )
        self.model = None
        self.vectorizer = None
        self._load_model()

    def _load_model(self):
        model_path = os.path.join(self.model_dir, self.MODEL_FILENAME)
        vec_path = os.path.join(self.model_dir, self.VECTORIZER_FILENAME)

        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
            except Exception:
                pass

        if os.path.exists(vec_path):
            try:
                self.vectorizer = joblib.load(vec_path)
            except Exception:
                pass

    def _get_file_size_category(self, size: int) -> str:
        if size < 1024:
            return "tiny"
        elif size < 10 * 1024:
            return "small"
        elif size < 100 * 1024:
            return "medium"
        elif size < 1024 * 1024:
            return "large"
        elif size < 10 * 1024 * 1024:
            return "huge"
        return "massive"

    def extract_text(self, filepath: str) -> str:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()[:100000]
        except Exception:
            return ""

    def classify(
        self, filepath: str, file_size: int = None
    ) -> Tuple[Optional[str], float, str]:
        raise NotImplementedError


class ContentClassifier:
    SUPPORTED_EXTENSIONS = {
        ".txt", ".md", ".log", ".json", ".xml", ".csv", ".py", ".js", ".java", 
        ".cpp", ".c", ".h", ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".html",
        ".css", ".ts", ".jsx", ".tsx", ".sql", ".sh", ".bat", ".ps1",
        ".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp",
        ".pdf",
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp",
        ".zip", ".rar", ".7z", ".tar", ".gz",
    }

    TEXT_EXTENSIONS = {
        ".txt", ".md", ".log", ".json", ".xml", ".csv", ".py", ".js", ".java",
        ".cpp", ".c", ".h", ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".html",
        ".css", ".ts", ".jsx", ".tsx", ".sql", ".sh", ".bat", ".ps1",
    }

    OFFICE_EXTENSIONS = {".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp"}

    PDF_EXTENSIONS = {".pdf"}

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

    ARCHIVE_EXTENSIONS = {".zip", ".rar", ".7z", ".tar", ".gz"}

    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or os.path.join(
            os.path.dirname(__file__), "../models/content"
        )
        self.text_classifier = None
        self.office_classifier = None
        self.pdf_classifier = None
        self.image_classifier = None
        self.archive_classifier = None
        self._load_models()

    def _load_models(self):
        try:
            from .text_classifier import TextClassifier
            self.text_classifier = TextClassifier(self.model_dir)
        except Exception as e:
            logger.warning(f"TextClassifier not loaded: {e}")

        try:
            from .office_classifier import OfficeClassifier
            self.office_classifier = OfficeClassifier(self.model_dir)
        except Exception as e:
            logger.warning(f"OfficeClassifier not loaded: {e}")

        try:
            from .pdf_classifier import PdfClassifier
            self.pdf_classifier = PdfClassifier(self.model_dir)
        except Exception as e:
            logger.warning(f"PdfClassifier not loaded: {e}")

        try:
            from .image_classifier import ImageClassifier
            self.image_classifier = ImageClassifier(self.model_dir)
        except Exception as e:
            logger.warning(f"ImageClassifier not loaded: {e}")

        try:
            from .archive_classifier import ArchiveClassifier
            self.archive_classifier = ArchiveClassifier(self.model_dir)
        except Exception as e:
            logger.warning(f"ArchiveClassifier not loaded: {e}")

    def get_file_type(self, filepath: str) -> str:
        ext = os.path.splitext(filepath)[1].lower()
        if ext in self.TEXT_EXTENSIONS:
            return "text"
        elif ext in self.OFFICE_EXTENSIONS:
            return "office"
        elif ext in self.PDF_EXTENSIONS:
            return "pdf"
        elif ext in self.IMAGE_EXTENSIONS:
            return "image"
        elif ext in self.ARCHIVE_EXTENSIONS:
            return "archive"
        return "unknown"

    def classify(
        self, filepath: str, use_ml: bool = True
    ) -> Tuple[Optional[str], float, Optional[str]]:
        if not os.path.exists(filepath):
            return None, 0.0, "file_not_found"

        file_type = self.get_file_type(filepath)
        file_size = os.path.getsize(filepath)

        if use_ml:
            if file_type == "text" and self.text_classifier:
                return self.text_classifier.classify(filepath, file_size)
            elif file_type == "office" and self.office_classifier:
                return self.office_classifier.classify(filepath, file_size)
            elif file_type == "pdf" and self.pdf_classifier:
                return self.pdf_classifier.classify(filepath, file_size)
            elif file_type == "image" and self.image_classifier:
                return self.image_classifier.classify(filepath, file_size)
            elif file_type == "archive" and self.archive_classifier:
                return self.archive_classifier.classify(filepath, file_size)

        return self._classify_by_extension(filepath, file_type, file_size)

    def _classify_by_extension(
        self, filepath: str, file_type: str, file_size: int
    ) -> Tuple[Optional[str], float, str]:
        ext = os.path.splitext(filepath)[1].lower()
        
        category_map = {
            "text": "TextDocument",
            "office": "OfficeDocument",
            "pdf": "PDFDocument",
            "image": "Image",
            "archive": "Archive",
        }
        
        category = category_map.get(file_type, "Unknown")
        return category, 0.5, "extension"

    def verify_extension(
        self, filepath: str
    ) -> Tuple[bool, Optional[str]]:
        expected_ext = os.path.splitext(filepath)[1].lower()
        predicted_type, confidence, method = self.classify(filepath)

        if method == "extension":
            return True, None

        type_to_ext = {
            "TextDocument": self.TEXT_EXTENSIONS,
            "OfficeDocument": self.OFFICE_EXTENSIONS,
            "PDFDocument": self.PDF_EXTENSIONS,
            "Image": self.IMAGE_EXTENSIONS,
            "Archive": self.ARCHIVE_EXTENSIONS,
        }

        if predicted_type in type_to_ext:
            expected_set = type_to_ext[predicted_type]
            is_valid = expected_ext in expected_set
            return is_valid, predicted_type

        return True, predicted_type