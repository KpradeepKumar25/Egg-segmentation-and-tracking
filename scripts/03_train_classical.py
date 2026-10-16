"""Task 2: train the classical propose-and-rank segmenter on the DEV split only.
  python scripts/03_train_classical.py --work work
Writes work/task2/pipeline_models.joblib and final_config.json (frozen before Task 3 touches the eval split).
The full experiment log (T0-T5, including discarded trials) is in results/task2/experiment_log.md.
"""
import _common, argparse, hashlib, json, os, cv2, joblib, numpy as np, pandas as pd
from eggcv import classical
ap = argparse.ArgumentParser(); ap.add_argument("--work", default="work"); a = ap.parse_args()
W = a.work; man = pd.read_csv(f"{W}/splits/split_manifest.csv"); dev = man[man.split == "dev"]
imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in dev.path]
labs = [np.load(f"{W}/task1/labels/{s}.npy") for s in dev.stem]
res = classical.train_pipeline(imgs, labs, dev.group.tolist())
os.makedirs(f"{W}/task2", exist_ok=True); mp = f"{W}/task2/pipeline_models.joblib"; joblib.dump(res, mp)
cfg = dict(final_trial="T5", models_sha256=hashlib.sha256(open(mp, "rb").read()).hexdigest(), threshold=res["thr"], topk=res["topk"],
           features=res["features"], dev_oof=res["dev_oof"], n_dev_images=len(dev))
json.dump(cfg, open(f"{W}/task2/final_config_repro.json", "w"), indent=2, default=float); print(json.dumps(cfg["dev_oof"], indent=1, default=float))
