"""Download/cache the default historical dataset without running models."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from garch_var.data import download_moex

path, manifest = download_moex(root)
print(path)
print(manifest["quality"])
