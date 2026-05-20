import unittest
import os
import sys
import tempfile
import shutil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from cli import cmd_predict, cmd_stats

class Args:
    pass

class TestCLI(unittest.TestCase):
    def test_predict(self):
        args = Args()
        args.filename = "invoice_2024-03-15.pdf"
        args.model = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/tfidf_logreg.joblib'))
        # Проверяем, что не возникает исключений
        cmd_predict(args)

    def test_stats(self):
        # Создаём временную структуру папок
        tmpdir = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(tmpdir, "Invoices"), exist_ok=True)
            with open(os.path.join(tmpdir, "Invoices", "test.pdf"), "w") as f:
                f.write("test")
            args = Args()
            args.path = tmpdir
            cmd_stats(args)
        finally:
            shutil.rmtree(tmpdir)

if __name__ == "__main__":
    unittest.main()
