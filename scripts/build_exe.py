import PyInstaller.__main__
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PyInstaller.__main__.run([
    "--onedir",
    "--windowed",
    "--noconfirm",
    "--name=FileSorter",
    "--distpath", os.path.join(ROOT, "dist"),
    "--workpath", os.path.join(ROOT, "build"),
    "--paths", ROOT,
    "--hidden-import=src._venv_setup",
    "--hidden-import=configs.config",
    "--hidden-import=src.inference.predict",
    "--hidden-import=src.data_prep.dataset",
    "--hidden-import=src.models.train",
    "--hidden-import=src.models.model",
    "--hidden-import=sklearn",
    "--hidden-import=sklearn.ensemble",
    "--hidden-import=sklearn.linear_model",
    "--hidden-import=sklearn.tree",
    "--hidden-import=sklearn.preprocessing",
    "--hidden-import=sklearn.pipeline",
    "--hidden-import=sklearn.metrics",
    "--hidden-import=sklearn.model_selection",
    "--hidden-import=sklearn.feature_extraction",
    "--hidden-import=pandas",
    "--hidden-import=numpy",
    "--hidden-import=joblib",
    "--hidden-import=ttkbootstrap",
    "--collect-all=ttkbootstrap",
    "--collect-all=sklearn",
    os.path.join(ROOT, "scripts", "gui.py"),
])
