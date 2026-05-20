import os
import joblib
from typing import Tuple, Optional
from PIL import Image
from sklearn.linear_model import LogisticRegression
import logging

from . import BaseContentClassifier

logger = logging.getLogger(__name__)


class ImageClassifier(BaseContentClassifier):
    CATEGORIES = [
        "Photo",
        "Screenshot",
        "Document",
        "Artwork",
        "Diagram",
    ]

    MODEL_SUBDIR = "image"

    def __init__(self, model_dir: str = None):
        subdir = os.path.join(model_dir or os.path.join(
            os.path.dirname(__file__), "../../models/content"
        ), self.MODEL_SUBDIR)
        super().__init__(subdir)

    def _load_model(self):
        model_path = os.path.join(self.model_dir, "model.joblib")
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
            except Exception:
                self.model = None

    def extract_features(self, filepath: str) -> dict:
        features = {}
        
        try:
            with Image.open(filepath) as img:
                width, height = img.size
                features["width"] = width
                features["height"] = height
                features["aspect_ratio"] = width / height if height > 0 else 0
                features["mode"] = img.mode
                
                try:
                    exif = img.getexif()
                    if exif:
                        features["has_exif"] = True
                        features["has_gps"] = "GPSInfo" in exif
                except Exception:
                    features["has_exif"] = False
                    features["has_gps"] = False

        except Exception as e:
            logger.warning(f"Failed to extract image features from {filepath}: {e}")
            return {"error": str(e)}

        return features

    def classify(
        self, filepath: str, file_size: int = None
    ) -> Tuple[Optional[str], float, str]:
        if file_size is None:
            file_size = os.path.getsize(filepath)

        features = self.extract_features(filepath)
        
        if "error" in features:
            return None, 0.0, "read_error"

        if self.model:
            try:
                feature_vector = [
                    features.get("width", 0),
                    features.get("height", 0),
                    features.get("aspect_ratio", 0),
                ]
                prediction = self.model.predict([feature_vector])[0]
                return prediction, 0.7, "ml"
            except Exception:
                pass

        return self._rule_based_classify(features, file_size)

    def _rule_based_classify(
        self, features: dict, file_size: int
    ) -> Tuple[Optional[str], float, str]:
        aspect_ratio = features.get("aspect_ratio", 1.0)
        has_exif = features.get("has_exif", False)
        has_gps = features.get("has_gps", False)
        mode = features.get("mode", "")

        width = features.get("width", 0)
        height = features.get("height", 0)

        if width > 0 and height > 0:
            if aspect_ratio > 1.9 and aspect_ratio < 2.1:
                if width >= 1920 and (has_exif or has_gps):
                    return "Photo", 0.8, "rules"
                return "Screenshot", 0.7, "rules"

            if aspect_ratio > 0.5 and aspect_ratio < 0.52:
                if height >= 1080 and (has_exif or has_gps):
                    return "Photo", 0.8, "rules"

        if has_exif and (has_gps or mode in ["RGB", "RGBA"]):
            return "Photo", 0.6, "rules"

        if mode in ["P", "LA"]:
            return "Artwork", 0.5, "mode"

        size_category = self._get_file_size_category(file_size)
        if size_category in ["tiny", "small"]:
            if aspect_ratio > 0.7 and aspect_ratio < 1.5:
                return "Diagram", 0.5, "size"

        return "Image", 0.5, "default"


def train_image_classifier(data_dir: str, output_dir: str):
    features = []
    labels = []

    from PIL import Image
    
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

    for filename in os.listdir(data_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in image_extensions:
            continue

        filepath = os.path.join(data_dir, filename)
        if os.path.isfile(filepath):
            clf = ImageClassifier()
            try:
                feat = clf.extract_features(filepath)
                if "error" not in feat:
                    features.append([
                        feat.get("width", 0),
                        feat.get("height", 0),
                        feat.get("aspect_ratio", 0),
                    ])
                    
                    if "photo" in filename.lower():
                        label = "Photo"
                    elif "screenshot" in filename.lower() or "screen" in filename.lower():
                        label = "Screenshot"
                    elif "doc" in filename.lower():
                        label = "Document"
                    elif "art" in filename.lower():
                        label = "Artwork"
                    else:
                        label = "Image"
                    
                    labels.append(label)
            except Exception:
                continue

    if not features:
        return False

    clf = LogisticRegression()
    clf.fit(features, labels)

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(output_dir, "model.joblib"))

    return True