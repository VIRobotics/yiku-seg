from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
del ROOT

__version__ = "0.2.0+xpu"
