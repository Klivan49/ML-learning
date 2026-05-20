import os
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from filename_parser import FileNameParser


def load_data(path):
    df = pd.read_csv(path)
    return df["filename"].tolist(), df["category"].tolist()


def preprocess_filenames(filenames):
    result = []
    for f in filenames:
        tokens = FileNameParser.tokenize(f)
        ext = os.path.splitext(f)[1].lower()
        if ext:
            tokens.append(ext)
        result.append(" ".join(tokens))
    return result


def train_model(X, y, out_model_path):
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=200, max_depth=20)
    clf.fit(X_vec, y)
    joblib.dump({"vectorizer": vectorizer, "clf": clf}, out_model_path)
    return clf, vectorizer


def evaluate_model(clf, vectorizer, X, y):
    X_vec = vectorizer.transform(X)
    y_pred = clf.predict(X_vec)
    acc = accuracy_score(y, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y, y_pred, average="weighted")
    cm = confusion_matrix(y, y_pred, labels=clf.classes_)
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {pr:.3f}, Recall: {rc:.3f}, F1: {f1:.3f}")
    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "../data/dataset.csv")
    model_path = os.path.join(
        os.path.dirname(__file__), "../models/tfidf_logreg.joblib"
    )
    X, y = load_data(data_path)
    X_proc = preprocess_filenames(X)
    clf, vectorizer = train_model(X_proc, y, model_path)
    evaluate_model(clf, vectorizer, X_proc, y)
    print(f"Model saved to {model_path}")
