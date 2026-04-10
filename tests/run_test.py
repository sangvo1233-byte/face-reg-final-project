"""
Test Runner — chay test va ghi ket qua ra file.
"""
import sys
import io
import os

# Redirect stdout to both console and file
class TeeWriter:
    def __init__(self, *writers):
        self.writers = writers
    def write(self, data):
        for w in self.writers:
            w.write(data)
            w.flush()
    def flush(self):
        for w in self.writers:
            w.flush()

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, '.')

    outfile = open('tests/test_result.txt', 'w', encoding='utf-8')
    sys.stdout = TeeWriter(sys.__stdout__, outfile)

    from tests.test_core import test_full_pipeline
    success = test_full_pipeline()

    outfile.close()
    sys.stdout = sys.__stdout__
    sys.exit(0 if success else 1)
