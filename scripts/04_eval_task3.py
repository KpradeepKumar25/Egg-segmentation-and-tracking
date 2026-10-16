"""Task 3: ONE-SHOT evaluation of the frozen classical segmenter on the held-out eval split vs the Task 1 masks.
  python scripts/04_eval_task3.py --work work
Writes work/task3/{pred/*.npy, eval_per_image.csv, eval_per_egg.csv, eval_metrics.json, analysis_plots.png, worst_cases.png, best_cases.png}
"""
import _common, argparse, json, os, cv2, joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from eggcv import classical, metrics
ap = argparse.ArgumentParser(); ap.add_argument("--work", default="work"); a = ap.parse_args()
W = a.work; man = pd.read_csv(f"{W}/splits/split_manifest.csv"); ev = man[man.split == "eval"]
b = joblib.load(f"{W}/task2/pipeline_models.joblib"); os.makedirs(f"{W}/task3/pred", exist_ok=True)
imgs, gts, prs = [], [], []
for r in ev.itertuples():
    img = cv2.cvtColor(cv2.imread(r.path), cv2.COLOR_BGR2RGB); gt = np.load(f"{W}/task1/labels/{r.stem}.npy")
    pf = f"{W}/task3/pred/{r.stem}.npy"
    if os.path.exists(pf): pr = np.load(pf)
    else: pr = classical.segment_final(img, b["m1"], b["m2"], b["thr"], b["topk"]); np.save(pf, pr)
    imgs.append(img); gts.append(gt); prs.append(pr)
per, eggs = metrics.per_image_table(ev.stem.tolist(), gts, prs); per.to_csv(f"{W}/task3/eval_per_image.csv", index=False); eggs.to_csv(f"{W}/task3/eval_per_egg.csv", index=False)
res = metrics.summarise(per); ci = metrics.bootstrap_ci(per); tax = eggs.category.value_counts()
json.dump(dict(eval=res, ci={k: list(v) for k, v in ci.items()}, taxonomy=tax.to_dict(), n_images=len(per), n_gt=int(per.n_gt.sum())), open(f"{W}/task3/eval_metrics.json", "w"), indent=2, default=float)
print(pd.Series(res).round(3).to_string()); print({k: tuple(round(x, 3) for x in v) for k, v in ci.items()}); print(tax.to_string())

def contour_vis(img, gt, pr):
    v = img.copy()
    for m, c in [(gt > 0, (0, 255, 0)), (pr > 0, (255, 0, 0))]:
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(v, cs, -1, c, 3)
    return v
rng = np.random.default_rng(0)
fig, ax = plt.subplots(1, 4, figsize=(24, 5))
ax[0].hist(per.fg_iou.dropna(), bins=np.linspace(0, 1, 11)); ax[0].set_title("per-image foreground IoU")
ax[1].scatter(eggs.area_frac, eggs.iou); ax[1].axhline(.5, ls="--", c="gray"); ax[1].set_title("does egg size matter?"); ax[1].set_xlabel("egg area / image area"); ax[1].set_ylabel("IoU")
tax.sort_values().plot.barh(ax=ax[2]); ax[2].set_title("failure taxonomy (per GT egg)")
ax[3].scatter(per.n_gt + rng.normal(0, .05, len(per)), per.n_pred + rng.normal(0, .05, len(per))); mx = max(per.n_gt.max(), per.n_pred.max()) + 1; ax[3].plot([0, mx], [0, mx], "k--"); ax[3].set_title("count agreement")
plt.tight_layout(); plt.savefig(f"{W}/task3/analysis_plots.png", dpi=130); plt.close()
idx = {s: i for i, s in enumerate(ev.stem)}
def grid(stems, title, path):
    cols = 4; rows = int(np.ceil(len(stems) / cols)); fig, axs = plt.subplots(rows, cols, figsize=(22, 5.6 * rows)); axs = np.atleast_1d(axs).ravel()
    for x in axs: x.axis("off")
    for x, s in zip(axs, stems):
        i = idx[s]; r = per.set_index("stem").loc[s]; x.imshow(contour_vis(imgs[i], gts[i], prs[i])); x.set_title(f"{s[:24]} GT={int(r.n_gt)} pred={int(r.n_pred)} fgIoU={r.fg_iou:.2f}", fontsize=10)
    plt.suptitle(title + "  green = Task 1 mask, red = classical prediction", fontsize=15); plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()
srt = per.assign(k=per.fg_iou.fillna(1.0)).sort_values("k")
grid(srt.stem.head(8).tolist(), "WORST 8 eval images", f"{W}/task3/worst_cases.png"); grid(srt.stem.tail(4).tolist(), "BEST 4 eval images", f"{W}/task3/best_cases.png")
