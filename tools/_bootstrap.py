"""Lets a script in tools/ import the chop package from the repository root.

Python puts the script's own folder (tools/) on the import path, not the root. Run tools from
the repository root (run.bat does), since results/, runs/ and check_frames/ are relative paths:

    run.bat tools\\memtest.py --minutes 20
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
