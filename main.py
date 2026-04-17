"""Entry point that delegates to the scanner package under src/."""

import runpy
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))
runpy.run_path(str(SRC / "main.py"), run_name="__main__")
