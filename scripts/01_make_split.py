"""Download the dataset (Roboflow egg-vzhtr v2) and make the leak-free 4:1 dev/eval split.
  export ROBOFLOW_API_KEY=...        # never commit the key
  python scripts/01_make_split.py --work work
"""
import _common, argparse, os
from eggcv import data
ap = argparse.ArgumentParser(); ap.add_argument("--work", default="work"); ap.add_argument("--raw", default=None, help="already-downloaded Roboflow folder export (skips download)")
a = ap.parse_args()
raw = a.raw or data.download_roboflow(os.environ["ROBOFLOW_API_KEY"], f"{a.work}/egg_raw")
m = data.group_split(data.build_manifest(raw, f"{a.work}/images"))
os.makedirs(f"{a.work}/splits", exist_ok=True); m.to_csv(f"{a.work}/splits/split_manifest.csv", index=False)
print(m.split.value_counts().to_string(), "\ngroups:", m.group.nunique(), "| no group in both splits: OK")
