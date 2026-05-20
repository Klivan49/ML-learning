import unittest
import tempfile
import os
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from watcher import DownloadEventHandler
from watchdog.events import FileCreatedEvent

class TestDownloadEventHandler(unittest.TestCase):
    def setUp(self):
        self.handler = DownloadEventHandler()
        self.tempfile = tempfile.NamedTemporaryFile(delete=False)
        self.tempfile.close()

    def tearDown(self):
        try:
            os.unlink(self.tempfile.name)
        except FileNotFoundError:
            pass

    def test_on_created(self):
        event = FileCreatedEvent(self.tempfile.name)
        # Проверяем, что не возникает исключений
        try:
            self.handler.on_created(event)
        except Exception as e:
            self.fail(f"on_created raised Exception: {e}")

if __name__ == "__main__":
    unittest.main()
