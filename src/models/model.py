from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from configs.config import CONFIG
from src.features.features import get_feature_columns, FILENAME_TEXT_COL, CONTENT_TEXT_COL


def _build_column_transformer(with_scaler: bool = True) -> ColumnTransformer:
    num_cols = get_feature_columns()
    scaler = StandardScaler() if with_scaler else "passthrough"

    return ColumnTransformer([
        ("num", scaler, num_cols),
        ("name_tfidf", TfidfVectorizer(**CONFIG.tfidf_filename_params), FILENAME_TEXT_COL),
        ("content_tfidf", TfidfVectorizer(**CONFIG.tfidf_content_params), CONTENT_TEXT_COL),
    ])


def build_logistic_regression() -> Pipeline:
    params = CONFIG.model_params["logistic_regression"]
    return Pipeline([
        ("features", _build_column_transformer(with_scaler=True)),
        ("clf", LogisticRegression(**params)),
    ])


def build_random_forest() -> Pipeline:
    params = CONFIG.model_params["random_forest"]
    return Pipeline([
        ("features", _build_column_transformer(with_scaler=False)),
        ("clf", RandomForestClassifier(**params)),
    ])


def build_gradient_boosting() -> Pipeline:
    params = CONFIG.model_params["gradient_boosting"]
    return Pipeline([
        ("features", _build_column_transformer(with_scaler=False)),
        ("clf", GradientBoostingClassifier(**params)),
    ])


MODEL_REGISTRY = {
    "logistic_regression": build_logistic_regression,
    "random_forest": build_random_forest,
    "gradient_boosting": build_gradient_boosting,
}
