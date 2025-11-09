"""Refresh the vendored CSV only when its pinned upstream checksum matches."""
from pathlib import Path
import hashlib
import json
import urllib.request

root = Path(__file__).resolve().parents[1]
source = json.loads((root / "datasets/provenance.json").read_text())
with urllib.request.urlopen(source["url"], timeout=30) as response:
    data = response.read()
if hashlib.sha256(data).hexdigest() != source["sha256"]:
    raise ValueError("Upstream data checksum mismatch")
(root / "datasets/robyn_weekly.csv").write_bytes(data)
