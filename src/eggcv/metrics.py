"""Instance-segmentation metrics and failure taxonomy (Tasks 2 and 3).

Ground truth = Task 1 masks, prediction = classical masks. Both are integer label maps (0 background, 1..N instances).
Instances are matched one-to-one with the Hungarian algorithm on IoU; a match counts as a true positive if IoU >= threshold.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def match_instances(gt, pr):
    """Hungarian matching on IoU. Returns ([(gt_idx, pred_idx, iou)], n_gt, n_pred)."""
    g, p = int(gt.max()), int(pr.max())
    if g == 0 or p == 0: return [], g, p
    pair = np.bincount((gt.ravel() * (p + 1) + pr.ravel()), minlength=(g + 1) * (p + 1)).reshape(g + 1, p + 1)
    inter = pair[1:, 1:].astype(float)
    ga, pa = pair[1:, :].sum(1), pair[:, 1:].sum(0)
    iou = inter / (ga[:, None] + pa[None, :] - inter + 1e-9)
    r, c = linear_sum_assignment(-iou)
    return [(i, j, iou[i, j]) for i, j in zip(r, c)], g, p


def evaluate(pairs, thr=0.5):
    """pairs: iterable of (gt_label_map, pred_label_map). Dataset-level precision / recall / F1, matched IoU, foreground IoU / Dice, count MAE."""
    TP = FP = FN = 0; ious, fg_iou, fg_dice, cnt = [], [], [], []
    for gt, pr in pairs:
        m, g, p = match_instances(gt, pr)
        tp = sum(1 for _, _, v in m if v >= thr); TP += tp; FP += p - tp; FN += g - tp
        ious += [v for _, _, v in m if v >= thr]
        a, b = gt > 0, pr > 0
        if a.any():
            fg_iou.append((a & b).sum() / ((a | b).sum() + 1e-9)); fg_dice.append(2 * (a & b).sum() / (a.sum() + b.sum() + 1e-9))
        cnt.append(abs(int(g) - int(p)))
    prec, rec = TP / max(TP + FP, 1), TP / max(TP + FN, 1)
    return dict(precision=prec, recall=rec, F1_at_0_5=2 * prec * rec / max(prec + rec, 1e-9), mean_inst_IoU=float(np.mean(ious)) if ious else 0.0,
                fg_IoU=float(np.mean(fg_iou)), fg_Dice=float(np.mean(fg_dice)), count_MAE=float(np.mean(cnt)))


def analyse_image(gt, pr):
    """Per-image analysis: IoU of every GT egg and a failure category for it, plus the number of false-positive regions."""
    g, p = int(gt.max()), int(pr.max())
    ga = np.array([(gt == k).sum() for k in range(1, g + 1)], float); pa = np.array([(pr == k).sum() for k in range(1, p + 1)], float)
    inter = np.zeros((g, p)); iou = np.zeros((g, p))
    if g and p:
        pair = np.bincount(gt.ravel() * (p + 1) + pr.ravel(), minlength=(g + 1) * (p + 1)).reshape(g + 1, p + 1)
        inter = pair[1:, 1:].astype(float); iou = inter / (ga[:, None] + pa[None, :] - inter + 1e-9)
    match = {}
    if g and p:
        r, c = linear_sum_assignment(-iou); match = {int(i): int(j) for i, j in zip(r, c) if iou[i, j] >= 0.5}
    cats, ious = [], []
    for i in range(g):
        if i in match:
            v = iou[i, match[i]]; ious.append(v); cats.append("good (IoU>=0.75)" if v >= 0.75 else "boundary error (0.5<=IoU<0.75)"); continue
        ious.append(iou[i].max() if p else 0.0)
        if p == 0: cats.append("miss (nothing predicted)"); continue
        j = int(np.argmax(inter[i])); cover = inter[i, j] / ga[i]; prec = inter[i, j] / pa[j]
        if (inter[:, j] / ga > 0.3).sum() >= 2: cats.append("merge (one region covers 2+ eggs)")
        elif (inter[i] / ga[i] > 0.2).sum() >= 2: cats.append("split (egg broken into 2+ regions)")
        elif cover < 0.1: cats.append("miss (wrong object chosen)")
        elif cover >= 0.5 and prec < 0.5: cats.append("leak (region far larger than egg)")
        elif prec >= 0.5 and cover < 0.5: cats.append("partial (only part of the egg)")
        else: cats.append("mislocalised")
    matched_preds = set(match.values()); fp = 0
    for j in range(p):
        if j in matched_preds: continue
        if g == 0 or inter[:, j].sum() / pa[j] < 0.3: fp += 1
    return dict(g=g, p=p, ious=ious, cats=cats, fp=fp, ga=ga)


def per_image_table(stems, gts, prs):
    """DataFrames used for the Task 3 report: one row per image and one row per GT egg."""
    rows, egg_rows = [], []
    for s, gt, pr in zip(stems, gts, prs):
        a = analyse_image(gt, pr); A, B = gt > 0, pr > 0
        rows.append(dict(stem=s, n_gt=a["g"], n_pred=a["p"], fg_iou=(A & B).sum() / ((A | B).sum() + 1e-9) if A.any() else np.nan,
                         fg_dice=2 * (A & B).sum() / (A.sum() + B.sum() + 1e-9) if A.any() else np.nan, false_pos=a["fp"],
                         tp50=sum(v >= .5 for v, c in zip(a["ious"], a["cats"]) if c.startswith(("good", "boundary"))), tp75=sum(v >= .75 for v in a["ious"])))
        for k, (v, c) in enumerate(zip(a["ious"], a["cats"])): egg_rows.append(dict(stem=s, egg=k + 1, iou=v, category=c, area_frac=a["ga"][k] / gt.size))
    return pd.DataFrame(rows), pd.DataFrame(egg_rows)


def summarise(per):
    G, P = per.n_gt.sum(), per.n_pred.sum(); tp, tp75 = per.tp50.sum(), per.tp75.sum()
    pr_, rc_ = tp / max(P, 1), tp / max(G, 1); pr75, rc75 = tp75 / max(P, 1), tp75 / max(G, 1); has = per.n_gt > 0
    return dict(precision=pr_, recall=rc_, F1_at_0_5=2 * pr_ * rc_ / max(pr_ + rc_, 1e-9), F1_at_0_75=2 * pr75 * rc75 / max(pr75 + rc75, 1e-9),
                fg_IoU=per.fg_iou[has].mean(), fg_Dice=per.fg_dice[has].mean(), count_MAE=(per.n_gt - per.n_pred).abs().mean(),
                exact_count_acc=(per.n_gt == per.n_pred).mean(), false_positive_regions=int(per.false_pos.sum()))


def bootstrap_ci(per, keys=("F1_at_0_5", "fg_IoU", "fg_Dice"), n=1000, seed=0):
    rng = np.random.default_rng(seed); boots = [summarise(per.iloc[rng.integers(0, len(per), len(per))].reset_index(drop=True)) for _ in range(n)]
    return {k: (float(np.percentile([b[k] for b in boots], 2.5)), float(np.percentile([b[k] for b in boots], 97.5))) for k in keys}
