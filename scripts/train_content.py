import os
import sys
import argparse
import glob
import logging
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from content import (
    TextClassifier,
    OfficeClassifier,
    PdfClassifier,
    ImageClassifier,
    ArchiveClassifier,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def prepare_text_data(data_dir: str):
    texts = []
    labels = []
    file_paths = []

    patterns = {
        "Study": ["*.txt", "*.md", "*.log", "*.py", "*.js", "*.java"],
        "Code": ["*.py", "*.js", "*.java", "*.cpp", "*.c", "*.h", "*.cs"],
        "Config": ["*.json", "*.xml", "*.yaml", "*.yml", "*.ini", "*.toml"],
        "Data": ["*.csv", "*.sql", "*.db"],
    }

    for category, globs in patterns.items():
        for pattern in globs:
            for filepath in glob.glob(os.path.join(data_dir, category, pattern)):
                if os.path.isfile(filepath):
                    try:
                        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                        texts.append(text[:50000])
                        labels.append(category)
                        file_paths.append(filepath)
                    except Exception as e:
                        logger.warning(f"Failed to read {filepath}: {e}")

    for filepath in glob.glob(os.path.join(data_dir, "*.txt")):
        if os.path.isfile(filepath) and filepath not in file_paths:
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                texts.append(text[:50000])
                labels.append("NotStudy")
                file_paths.append(filepath)
            except Exception:
                pass

    return texts, labels


def train_text_model(data_dir: str, output_dir: str):
    logger.info("Training text classifier...")
    texts, labels = prepare_text_data(data_dir)

    if not texts:
        logger.warning("No text data found")
        return False

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(X, labels)

    y_pred = clf.predict(X)
    acc = accuracy_score(labels, y_pred)
    logger.info(f"Text classifier accuracy: {acc:.3f}")

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(output_dir, "model.joblib"))
    joblib.dump(vectorizer, os.path.join(output_dir, "vectorizer.joblib"))
    logger.info(f"Text model saved to {output_dir}")

    return True


def prepare_office_data(data_dir: str):
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

                label = "NotStudy"
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

                labels.append(label)

    return texts, labels


def train_office_model(data_dir: str, output_dir: str):
    logger.info("Training office classifier...")
    texts, labels = prepare_office_data(data_dir)

    if not texts:
        logger.warning("No office data found")
        return False

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(X, labels)

    y_pred = clf.predict(X)
    acc = accuracy_score(labels, y_pred)
    logger.info(f"Office classifier accuracy: {acc:.3f}")

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(output_dir, "model.joblib"))
    joblib.dump(vectorizer, os.path.join(output_dir, "vectorizer.joblib"))
    logger.info(f"Office model saved to {output_dir}")

    return True


def prepare_pdf_data(data_dir: str):
    from content.pdf_classifier import extract_text_from_pdf

    texts = []
    labels = []

    for filename in os.listdir(data_dir):
        if not filename.lower().endswith(".pdf"):
            continue

        filepath = os.path.join(data_dir, filename)
        if os.path.isfile(filepath):
            text = extract_text_from_pdf(filepath)
            if text:
                texts.append(text[:50000])

                label = "NotStudy"
                if "study" in filename.lower() or "учеб" in filename.lower():
                    label = "Study"
                elif "invoice" in filename.lower() or "счёт" in filename.lower():
                    label = "Invoice"
                elif "report" in filename.lower() or "отчёт" in filename.lower():
                    label = "Report"
                elif "book" in filename.lower():
                    label = "Book"
                elif "manual" in filename.lower():
                    label = "Manual"

                labels.append(label)

    return texts, labels


def train_pdf_model(data_dir: str, output_dir: str):
    logger.info("Training PDF classifier...")
    texts, labels = prepare_pdf_data(data_dir)

    if not texts:
        logger.warning("No PDF data found")
        return False

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(X, labels)

    y_pred = clf.predict(X)
    acc = accuracy_score(labels, y_pred)
    logger.info(f"PDF classifier accuracy: {acc:.3f}")

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(output_dir, "model.joblib"))
    joblib.dump(vectorizer, os.path.join(output_dir, "vectorizer.joblib"))
    logger.info(f"PDF model saved to {output_dir}")

    return True


def prepare_image_data(data_dir: str):
    from PIL import Image

    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

    features = []
    labels = []

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

                    label = "Image"
                    if "photo" in filename.lower():
                        label = "Photo"
                    elif "screenshot" in filename.lower():
                        label = "Screenshot"
                    elif "doc" in filename.lower():
                        label = "Document"

                    labels.append(label)
            except Exception:
                continue

    return features, labels


def train_image_model(data_dir: str, output_dir: str):
    logger.info("Training image classifier...")
    features, labels = prepare_image_data(data_dir)

    if not features:
        logger.warning("No image data found")
        return False

    clf = LogisticRegression()
    clf.fit(features, labels)

    y_pred = clf.predict(features)
    acc = accuracy_score(labels, y_pred)
    logger.info(f"Image classifier accuracy: {acc:.3f}")

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(output_dir, "model.joblib"))
    logger.info(f"Image model saved to {output_dir}")

    return True


def prepare_archive_data(data_dir: str):
    archive_extensions = {".zip", ".rar", ".7z", ".tar", ".gz"}

    features = []
    labels = []

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

                label = "Other"
                if "backup" in filename.lower():
                    label = "Backup"
                elif "install" in filename.lower() or "setup" in filename.lower():
                    label = "Installer"
                elif "dataset" in filename.lower():
                    label = "Dataset"
                elif "project" in filename.lower():
                    label = "Project"

                labels.append(label)

    return features, labels


def train_archive_model(data_dir: str, output_dir: str):
    logger.info("Training archive classifier...")
    features, labels = prepare_archive_data(data_dir)

    if not features:
        logger.warning("No archive data found")
        return False

    clf = LogisticRegression()
    clf.fit(features, labels)

    y_pred = clf.predict(features)
    acc = accuracy_score(labels, y_pred)
    logger.info(f"Archive classifier accuracy: {acc:.3f}")

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(output_dir, "model.joblib"))
    logger.info(f"Archive model saved to {output_dir}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Train content classification models")
    parser.add_argument(
        "--data-dir", "-d",
        default="../data/content",
        help="Directory containing training data"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="../models/content",
        help="Output directory for models"
    )
    parser.add_argument(
        "--type", "-t",
        choices=["text", "office", "pdf", "image", "archive", "all"],
        default="all",
        help="Type of model to train"
    )

    args = parser.parse_args()

    data_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), args.data_dir)
    )
    output_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), args.output_dir)
    )

    results = {}

    if args.type in ["text", "all"]:
        results["text"] = train_text_model(
            os.path.join(data_dir, "text"), os.path.join(output_dir, "text")
        )

    if args.type in ["office", "all"]:
        results["office"] = train_office_model(
            os.path.join(data_dir, "office"), os.path.join(output_dir, "office")
        )

    if args.type in ["pdf", "all"]:
        results["pdf"] = train_pdf_model(
            os.path.join(data_dir, "pdf"), os.path.join(output_dir, "pdf")
        )

    if args.type in ["image", "all"]:
        results["image"] = train_image_model(
            os.path.join(data_dir, "image"), os.path.join(output_dir, "image")
        )

    if args.type in ["archive", "all"]:
        results["archive"] = train_archive_model(
            os.path.join(data_dir, "archive"), os.path.join(output_dir, "archive")
        )

    logger.info("Training complete!")
    for model_type, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {model_type}: {status}")


if __name__ == "__main__":
    main()