from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from configs.config import CONFIG


def build_logistic_regression() -> Pipeline:
    params = CONFIG.model_params["logistic_regression"]
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**params)),
    ])


def build_random_forest() -> Pipeline:
    params = CONFIG.model_params["random_forest"]
    return Pipeline([
        ("clf", RandomForestClassifier(**params)),
    ])


def build_gradient_boosting() -> Pipeline:
    params = CONFIG.model_params["gradient_boosting"]
    return Pipeline([
        ("clf", GradientBoostingClassifier(**params)),
    ])


MODEL_REGISTRY = {
    "logistic_regression": build_logistic_regression,
    "random_forest": build_random_forest,
    "gradient_boosting": build_gradient_boosting,
}
