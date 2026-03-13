"""Shared pytest fixtures and path setup."""
import sys
from pathlib import Path

# Ensure project root is on sys.path so both `src` and legacy modules resolve
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
