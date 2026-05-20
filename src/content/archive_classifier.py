import os
import re
import zipfile
import joblib
from typing import Tuple, Optional, List
from sklearn.linear_model import LogisticRegression
import logging

from . import BaseContentClassifier

logger = logging.getLogger(__name__)


class ArchiveClassifier(BaseContentClassifier):
    CATEGORIES = [
        "Backup",
        "Installer",
        "Dataset",
        "Project",
        "Other",
    ]

    MODEL_SUBDIR = "archive"

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

    def extract_file_list(self, filepath: str) -> List[str]:
        ext = os.path.splitext(filepath)[1].lower()
        files = []

        try:
            if ext in {".zip"}:
                with zipfile.ZipFile(filepath, "r") as zf:
                    files = zf.namelist()
            elif ext in {".rar", ".7z", ".tar", ".gz"}:
                logger.info(f"RAR/7z/TAR support requires external library")
        except Exception as e:
            logger.warning(f"Failed to extract file list from {filepath}: {e}")

        return files

    def extract_features(self, filepath: str, file_list: List[str]) -> dict:
        features = {}
        
        features["file_count"] = len(file_list)
        
        folder_count = sum(1 for f in file_list if f.endswith("/"))
        features["folder_count"] = folder_count
        
        extensions = set()
        for f in file_list:
            ext = os.path.splitext(f)[1].lower()
            if ext:
                extensions.add(ext)
        features["unique_extensions"] = len(extensions)
        features["extensions"] = extensions
        
        has_code = any(
            ext in {".py", ".js", ".java", ".cpp", ".c", ".h", ".cs", ".go", ".rs"}
            for ext in extensions
        )
        features["has_code"] = has_code
        
        has_docs = any(
            ext in {".txt", ".md", ".doc", ".docx", ".pdf"}
            for ext in extensions
        )
        features["has_docs"] = has_docs
        
        has_images = any(
            ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
            for ext in extensions
        )
        features["has_images"] = has_images
        
        has_data = any(
            ext in {".csv", ".json", ".xml", ".sql", ".db"}
            for ext in extensions
        )
        features["has_data"] = has_data

        root_files = [os.path.basename(f) for f in file_list if "/" not in f]
        features["root_file_count"] = len(root_files)
        
        backup_patterns = [r"backup", r"dump", r"save", r"db_", r"_backup", r"\.bak"]
        features["is_backup"] = any(
            re.search(p, f, re.IGNORECASE) for f in file_list for p in backup_patterns
        )
        
        installer_patterns = [r"setup", r"install", r"dist", r"build", r"release"]
        features["is_installer"] = any(
            re.search(p, f, re.IGNORECASE) for f in file_list for p in installer_patterns
        )
        
        dataset_patterns = [r"dataset", r"data", r"sample", r"train", r"test"]
        features["is_dataset"] = any(
            re.search(p, f, re.IGNORECASE) for f in file_list for p in dataset_patterns
        )
        
        project_patterns = [r"src", r"lib", r"config", r"package\.json"]
        features["is_project"] = any(
            re.search(p, f, re.IGNORECASE) for f in file_list for p in project_patterns
        )

        return features

    def classify(
        self, filepath: str, file_size: int = None
    ) -> Tuple[Optional[str], float, str]:
        if file_size is None:
            file_size = os.path.getsize(filepath)

        file_list = self.extract_file_list(filepath)
        if not file_list:
            return None, 0.0, "read_error"

        features = self.extract_features(filepath, file_list)

        if self.model:
            try:
                feature_vector = [
                    features.get("file_count", 0),
                    features.get("folder_count", 0),
                    features.get("unique_extensions", 0),
                    int(features.get("has_code", False)),
                    int(features.get("has_docs", False)),
                    int(features.get("has_data", False)),
                ]
                prediction = self.model.predict([feature_vector])[0]
                return prediction, 0.7, "ml"
            except Exception:
                pass

        return self._rule_based_classify(features, file_size)

    def _rule_based_classify(
        self, features: dict, file_size: int
    ) -> Tuple[Optional[str], float, str]:
        if features.get("is_backup"):
            return "Backup", 0.8, "rules"
        if features.get("is_installer"):
            return "Installer", 0.8, "rules"
        if features.get("is_dataset"):
            return "Dataset", 0.7, "rules"
        if features.get("is_project"):
            return "Project", 0.7, "rules"
        
        file_count = features.get("file_count", 0)
        unique_ext = features.get("unique_extensions", 0)
        
        if file_count > 100 and unique_ext > 10:
            return "Dataset", 0.6, "rules"
        
        if file_count == 1 and unique_ext == 1:
            return "Other", 0.5, "rules"

        size_category = self._get_file_size_category(file_size)
        if size_category in ["huge", "massive"]:
            if file_count > 50:
                return "Dataset", 0.5, "size"

        return "Other", 0.5, "default"


def train_archive_classifier(data_dir: str, output_dir: str):
    features = []
    labels = []

    archive_extensions = {".zip", ".rar", ".7z", ".tar", ".gz"}

    for filename in os.listdir(data_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in archive_extensions:
            continue

        filepath = os.path.join(data_dir, filename)
        if os.path.isfile(filepath):
            clf = ArchiveClassifier()
            file_list = clf.extract_file_list(filepath)
            if file_list:
                feat = clf.extract_features(filepath, file_list)
                features.append([
                    feat.get("file_count", 0),
                    feat.get("folder_count", 0),
                    feat.get("unique_extensions", 0),
                    int(feat.get("has_code", False)),
                    int(feat.get("has_docs", False)),
                    int(feat.get("has_data", False)),
                ])
                
                if "backup" in filename.lower():
                    label = "Backup"
                elif "install" in filename.lower() or "setup" in filename.lower():
                    label = "Installer"
                elif "dataset" in filename.lower() or "data" in filename.lower():
                    label = "Dataset"
                elif "project" in filename.lower():
                    label = "Project"
                else:
                    label = "Other"
                    
                labels.append(label)

    if not features:
        return False

    clf = LogisticRegression()
    clf.fit(features, labels)

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(output_dir, "model.joblib"))

    return True