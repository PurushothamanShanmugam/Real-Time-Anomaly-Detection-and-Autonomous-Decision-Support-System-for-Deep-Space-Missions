"""
tests/conftest.py
-----------------
Adds the project root to sys.path so that 'src', 'api', and 'kafka'
can be imported in all test files, regardless of how pytest is invoked
or where GitHub Actions clones the repository.
"""
import sys
from pathlib import Path

# Insert project root at the front of sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))