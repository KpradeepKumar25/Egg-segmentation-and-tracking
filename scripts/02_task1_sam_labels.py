"""Task 1: SAM + OWLv2 segmentation of all images, then the recorded human-review decisions.
  python scripts/02_task1_sam_labels.py --work work

Stage A (automatic): OWLv2 boxes -> SAM masks for all 160 images (strict, then fallback thresholds).
Stage B (review):    applies the human decisions recorded in eggcv.task1_sam.REVIEW_* (see README: the decisions are
                     indices into lists built in the same order; they match the original run. If your automatic output
                     differs slightly, use --no-review to keep the purely automatic labels).
Writes work/task1/labels/<stem>.npy (0 = background, 1..N = eggs), overlays/, task1_summary.csv.
"""
import _common, argparse, os, cv2, numpy as np, pandas as pd
from tqdm import tqdm
from eggcv import task1_sam as T

ap = argparse.ArgumentParser(); ap.add_argument("--work", default="work"); ap.add_argument("--ckpt", default="sam_vit_h_4b8939.pth"); ap.add_argument("--no-review", action="store_true")
a = ap.parse_args()
W = a.work; os.makedirs(f"{W}/task1/labels", exist_ok=True); os.makedirs(f"{W}/task1/overlays", exist_ok=True)
man = pd.read_csv(f"{W}/splits/split_manifest.csv"); path = dict(zip(man.stem, man.path))
rd = lambda s: cv2.cvtColor(cv2.imread(path[s]), cv2.COLOR_BGR2RGB)
M = T.Task1Models(a.ckpt)
lab_p = lambda s: f"{W}/task1/labels/{s}.npy"


def save(s, lbl):
    np.save(lab_p(s), lbl)
    cv2.imwrite(f"{W}/task1/overlays/{s}.jpg", cv2.cvtColor(T.overlay(rd(s), lbl), cv2.COLOR_RGB2BGR))


# ---- Stage A ----
rows = []
for r in tqdm(man.itertuples(), total=len(man)):
    lbl, kept, top, status = M.auto_label(rd(r.stem)); save(r.stem, lbl)
    rows.append(dict(stem=r.stem, split=r.split, n_eggs=len(kept), status=status, top=round(top, 3), min_det=round(min([k["det"] for k in kept], default=0.0), 3)))
t1 = pd.DataFrame(rows); print(t1.status.value_counts().to_string())
if a.no_review:
    t1.to_csv(f"{W}/task1/task1_summary.csv", index=False); raise SystemExit

# ---- Stage B: human review ----
def set_status(s, st, n):
    m = t1.stem == s
    t1.loc[m, "status"] = st; t1.loc[m, "n_eggs"] = n
none_stems = t1[t1.status == "none"].stem.tolist()
for i, picks in T.REVIEW_CHOICES.items():                       # B1: images where nothing was found; human picks a SAM-auto mask
    s = none_stems[i]; c = M.sam_candidates(rd(s)); img = rd(s); lbl = np.zeros(img.shape[:2], np.int32)
    for j, k in enumerate(picks, 1): lbl[T.clean_mask(c[k - 1]["seg"]) & (lbl == 0)] = j
    save(s, lbl); set_status(s, "manual_pick" if picks else "no_egg", len(picks))

rev = t1[(t1.status == "fallback") | ((t1.status == "strict") & (t1.n_eggs >= 2))].stem.tolist()
for i in sorted(set(T.REVIEW_REMOVE) | set(T.REVIEW_ADD)):       # B2: delete wrong eggs / add a missed egg by clicking on it
    s = rev[i]; lbl = np.load(lab_p(s))
    for k in T.REVIEW_REMOVE.get(i, []): lbl[lbl == k] = 0
    lbl = T.clean_relabel(lbl)
    for pt in T.REVIEW_ADD.get(i, []):
        seg = M.seg_from_points(rd(s), [pt]) & (lbl == 0); lbl[seg] = lbl.max() + 1
    save(s, lbl); set_status(s, "manual_fix", int(lbl.max()))

elig = t1[t1.status.isin(["strict", "fallback", "manual_fix"])].stem.tolist()   # B3: low-threshold proposals for missed eggs
props = {}
for s in tqdm(elig):
    p = M.propose_extra(rd(s), np.load(lab_p(s)))
    if p: props[s] = p
prop_stems = list(props)
strict1 = t1[(t1.status == "strict") & (t1.n_eggs == 1)].stem.tolist()
changed = []
for i, picks in T.REVIEW_ACCEPT.items():
    s = prop_stems[i]; lbl = np.load(lab_p(s))
    for k in picks:
        seg = props[s][k - 1]["seg"] & (lbl == 0); lbl[seg] = lbl.max() + 1
    save(s, lbl); changed.append(s)
for i, pts in T.REVIEW_EXTRA_POINTS.items():
    s = strict1[i]; lbl = np.load(lab_p(s))
    for pt in pts:
        seg = M.seg_from_points(rd(s), [pt]) & (lbl == 0); lbl[seg] = lbl.max() + 1
    save(s, lbl); changed.append(s)
for s in changed: set_status(s, "manual_add", int(np.load(lab_p(s)).max()))

# B4: one image where the egg is split by leaves: keep every SAM piece (holes filled)
sp = T.SPECIAL_BOX_FIX; s = strict1[sp["strict_index"]]
opts, _ = M.seg_from_box_options(rd(s), sp["box"]); save(s, opts[sp["sam_mask"]].astype(np.int32)); set_status(s, "manual_fix", 1)

# freeze: label files are the source of truth for counts
t1["n_eggs"] = [int(np.load(lab_p(s)).max()) for s in t1.stem]
t1.to_csv(f"{W}/task1/task1_summary.csv", index=False)
print(t1.status.value_counts().to_string(), "\ntotal GT eggs:", int(t1.n_eggs.sum()))
print(t1.groupby("split").agg(images=("stem", "count"), eggs=("n_eggs", "sum")).to_string())
