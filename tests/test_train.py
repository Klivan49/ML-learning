import unittest
import os
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))
from train import load_data, preprocess_filenames, train_model

class TestTrainScript(unittest.TestCase):
    def test_train_and_predict(self):
        # Мини-набор
        X = ["invoice_2023-01-01.pdf", "photo_2022-12-12.jpg", "project_v1.py"]
        y = ["Invoices", "Images", "Projects"]
        X_proc = preprocess_filenames(X)
        clf, vectorizer = train_model(X_proc, y, "/tmp/test_model.joblib")
        pred = clf.predict(vectorizer.transform(X_proc))
        self.assertEqual(list(pred), y)

if __name__ == "__main__":
    unittest.main()
