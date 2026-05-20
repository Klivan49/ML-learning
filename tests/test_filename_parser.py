import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from filename_parser import FileNameParser

class TestFileNameParser(unittest.TestCase):
    def test_tokenize(self):
        cases = {
            "Invoice_2023-12-01.pdf": ["invoice"],
            "myProjectReport_v2.docx": ["my", "project", "report", "v2"],
            "photo-20230411_123456.jpg": ["photo", "123456"],
            "archive_final-2022.zip": ["archive", "final"],
            "README.md": ["readme"]
        }
        for fname, expected in cases.items():
            tokens = FileNameParser.tokenize(fname)
            for token in expected:
                self.assertIn(token, tokens)

if __name__ == "__main__":
    unittest.main()
