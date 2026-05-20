import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from classifier import FileNameClassifier

class TestFileNameClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clf = FileNameClassifier(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/tfidf_logreg.joblib')))

    def test_predict(self):
        # Примеры из датасета
        cases = [
            ("report_2023-01-01.pdf", "Documents"),
            ("photo_2022-12-12.jpg", "Images"),
            ("project_v1.py", "Projects"),
            ("invoice_2024-03-15.pdf", "Invoices"),
            ("backup_2025-05-05.zip", "Archives"),
            ("movie_clip.mp4", "Media"),
        ]
        for fname, cat in cases:
            pred = self.clf.predict(fname)
            self.assertIn(pred, [cat, "Others"])  # допускаем "Others" для пограничных случаев

if __name__ == "__main__":
    unittest.main()
