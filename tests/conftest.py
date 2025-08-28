"""Pytest configuration helpers.

This file makes the project's `src/` directory importable during test
collection/execution so tests can `import etc` without requiring an
editable install.
"""
from pathlib import Path
import sys


# Insert the project's src folder (repo_root / 'src') at the front of sys.path
repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
sys.path.insert(0, str(src_path))
