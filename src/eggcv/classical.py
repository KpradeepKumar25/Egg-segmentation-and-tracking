"""Classical egg segmenter for photos (Task 2, final trial T5).

Propose-and-rank pipeline, no deep network:
  1. candidate_masks  : Otsu thresholds on L, a, b, S and a whiteness map, both polarities, morphology, hole filling
  2. cand_feats       : 25 shape / edge / colour / consensus features per candidate
  3. stage-1 ranker   : Random Forest predicts each candidate's IoU with the true egg
  4. add_refined      : GrabCut-refined versions of the top candidates are added to the pool
  5. stage-2 ranker   : second Random Forest re-ranks raw + refined candidates; the best is kept, others if score >= thr
"""
import cv2
import numpy as np
from scipy import ndimage as ndi

K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def whiteness(img):
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    w = L - 1.5 * np.sqrt(a ** 2 + b ** 2)
    return cv2.normalize(w, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


K9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
MIN_A = 0.02

def feature_channels(img):
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_RGB2LAB); hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
    return dict(L=lab[..., 0], a=lab[..., 1], b=lab[..., 2], S=hsv[..., 1], W=whiteness(img))

def candidate_masks(img):
    H, W = img.shape[:2]; out = []
    for name, ch in feature_channels(img).items():
        _, m = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for pol, mm in (("hi", m), ("lo", 255 - m)):
            mm = cv2.morphologyEx(cv2.morphologyEx(mm, cv2.MORPH_OPEN, K9), cv2.MORPH_CLOSE, K9)
            n, cc, st, _ = cv2.connectedComponentsWithStats((mm > 0).astype(np.uint8), 8)
            for i in range(1, n):
                if st[i, cv2.CC_STAT_AREA] >= MIN_A * H * W:
                    out.append((f"{name}_{pol}", ndi.binary_fill_holes(cc == i)))
    return out


S, R = 160, 320
CHN = [f"{c}_{p}" for c in ["L", "a", "b", "S", "W"] for p in ("hi", "lo")]
FEAT_NAMES = ["area", "solidity", "ellipse_fit", "border_frac", "edge", "contrast", "abs_contrast", "texture",
              "color_std", "center_dist", "mean_L", "chroma", "circularity", "consensus70", "consensus50"] + CHN
FEAT2 = FEAT_NAMES + ["refined"]; NF1 = len(FEAT_NAMES)
ds = lambda m, n=S: cv2.resize(m.astype(np.uint8), (n, n), interpolation=cv2.INTER_NEAREST).astype(bool)
pack = lambda m: np.packbits(m.ravel())
unpack = lambda p, shp: np.unpackbits(p)[:shp[0] * shp[1]].reshape(shp).astype(bool)
iou_b = lambda a, b: (a & b).sum() / ((a | b).sum() + 1e-9)

def cand_feats(comp_full, name, lab, grad):
    comp = ds(comp_full, R); H, W = comp.shape
    if comp.sum() < 50: return None
    cs, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea); ca = cv2.contourArea(c) + 1e-9; peri = cv2.arcLength(c, True)
    sol = ca / (cv2.contourArea(cv2.convexHull(c)) + 1e-9)
    ell = 0.0
    if len(c) >= 5:
        em = np.zeros((H, W), np.uint8); cv2.ellipse(em, cv2.fitEllipse(c), 1, -1); em = em.astype(bool); ell = iou_b(em, comp)
    bd = comp & ~ndi.binary_erosion(comp, iterations=1)
    onb = bd.copy(); onb[2:-2, 2:-2] = False; border_frac = onb.sum() / (bd.sum() + 1e-9)
    bdi = bd & ~onb; edge = grad[bdi].mean() if bdi.sum() > 15 else 0.0
    ring = ndi.binary_dilation(comp, iterations=6) & ~comp; L = lab[..., 0]
    contrast = L[comp].mean() - (L[ring].mean() if ring.any() else L[comp].mean())
    inner = ndi.binary_erosion(comp, iterations=3); inner = inner if inner.sum() > 30 else comp
    ys, xs = np.nonzero(comp); cdist = np.hypot(ys.mean() / H - .5, xs.mean() / W - .5)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)[comp].mean()
    return [comp.mean(), sol, ell, border_frac, edge, contrast, abs(contrast), grad[inner].mean(),
            lab[comp].std(0).mean(), cdist, L[comp].mean(), chroma, 4 * np.pi * ca / (peri ** 2 + 1e-9)]


# ---------------- reusable pipeline pieces (same code is used later for eval images and for the video) ----------------
def img_ctx(img):
    im_r = cv2.resize(img, (R, R), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(cv2.GaussianBlur(im_r, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]; grad = np.hypot(cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)) / 4.0
    return im_r, lab, grad

def rows_for(masks, names, lab, grad, refined):
    rows, sm, pk, nm = [], [], [], []
    for m, n in zip(masks, names):
        f = cand_feats(m, n, lab, grad)
        if f is None: continue
        rows.append(f + [0, 0] + [1.0 * (n == c) for c in CHN] + [float(refined)]); sm.append(ds(m)); pk.append(pack(m)); nm.append(n)
    return rows, sm, pk, nm

def consensus(X, small):
    if len(small) == 0: return X
    sm = np.stack(small).reshape(len(small), -1).astype(np.float32); inter = sm @ sm.T; ar = sm.sum(1)
    iou = inter / (ar[:, None] + ar[None, :] - inter + 1e-9); np.fill_diagonal(iou, 0)
    X[:, 13] = (iou > 0.7).sum(1); X[:, 14] = (iou > 0.5).sum(1); return X

def build_pool_stage1(img):
    im_r, lab, grad = img_ctx(img); ms = candidate_masks(img)
    rows, sm, pk, nm = rows_for([m for _, m in ms], [n for n, _ in ms], lab, grad, 0)
    X = consensus(np.array(rows, float).reshape(-1, len(FEAT2)), sm)
    return dict(X=np.nan_to_num(X), small=sm, packed=pk, names=nm, shape=img.shape[:2])

def grabcut_refine(im_r, m):
    m8 = m.astype(np.uint8); H, W = m.shape
    for k_in in (21, 11):
        sure = cv2.erode(m8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_in, k_in)))
        if sure.sum() >= 30: break
    else: return None
    near = cv2.dilate(m8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29)))
    mask = np.full((H, W), cv2.GC_BGD, np.uint8); mask[near > 0] = cv2.GC_PR_BGD; mask[m] = cv2.GC_PR_FGD; mask[sure > 0] = cv2.GC_FGD
    try: cv2.grabCut(im_r, mask, None, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error: return None
    out = ndi.binary_fill_holes((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(out, 8)
    if n < 2: return None
    return cc == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))

def add_refined(img, pool, scores1, topk=6):
    if len(pool["names"]) == 0: return pool
    im_r, lab, grad = img_ctx(img); H, W = img.shape[:2]; picks = []
    for i in np.argsort(-scores1):
        if all(iou_b(pool["small"][i], pool["small"][q]) < 0.85 for q in picks): picks.append(i)
        if len(picks) >= topk: break
    new = []
    for i in picks:
        g = grabcut_refine(im_r, ds(unpack(pool["packed"][i], pool["shape"]), R))
        if g is not None: new.append(cv2.resize(g.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) > 0.5)
    rows, sm, pk, nm = rows_for(new, ["gc"] * len(new), lab, grad, 1)
    X = np.vstack([pool["X"], np.array(rows, float).reshape(-1, len(FEAT2))]) if rows else pool["X"].copy()
    small = pool["small"] + sm
    return dict(X=np.nan_to_num(consensus(X, small)), small=small, packed=pool["packed"] + pk, names=pool["names"] + nm, shape=pool["shape"])

def select_from_pool(d, sc, thr, nms=0.3):
    out = np.zeros(d["shape"], np.int32); kept, k = [], 0
    for j, i in enumerate(np.argsort(-sc)):
        if j > 0 and sc[i] < thr: break
        if all(iou_b(d["small"][i], d["small"][q]) < nms for q in kept):
            kept.append(i); m = unpack(d["packed"][i], d["shape"]); free = m & (out == 0)
            if free.sum() >= MIN_A * m.size: k += 1; out[free] = k
    return out



def segment_final(img, m1, m2, thr, topk=6):
    """The complete classical pipeline: RGB image (H, W, 3) uint8 -> instance label map (0 background, 1..N eggs)."""
    p1 = build_pool_stage1(img)
    if len(p1["names"]) == 0: return np.zeros(img.shape[:2], np.int32)
    p2 = add_refined(img, p1, m1.predict(p1["X"][:, :NF1]), topk)
    return select_from_pool(p2, m2.predict(p2["X"]), thr)


# ---------------------------------------------------------------------------------------------------------------------
# Training (dev split only). Reproduces trial T5: two Random Forest stages, out-of-fold scores so no image sees its own label.
# ---------------------------------------------------------------------------------------------------------------------
def _targets(pool, gt):
    gts = [ds(gt == k) for k in range(1, int(gt.max()) + 1)]
    return np.array([max([iou_b(m, g) for g in gts], default=0.0) for m in pool["small"]])


def train_pipeline(images, labels, groups, n_estimators=300, n_splits=5, topk=6, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8), seed=0, verbose=True):
    """images: list of RGB uint8 arrays; labels: list of GT label maps; groups: near-duplicate group id per image (used for GroupKFold).
    Returns dict(m1, m2, thr, topk, dev_oof=metrics of the out-of-fold dev predictions at the chosen threshold)."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold
    from .metrics import evaluate
    mk = lambda: RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=3, n_jobs=-1, random_state=seed)
    n = len(images); folds = list(GroupKFold(min(n_splits, len(set(groups)))).split(np.arange(n), groups=groups))
    log = print if verbose else (lambda *a, **k: None)

    log("stage 1: candidate pools + features ...")
    P1 = [build_pool_stage1(im) for im in images]
    y1 = [_targets(p, g) if len(p["names"]) else np.zeros(0) for p, g in zip(P1, labels)]
    own = np.concatenate([[i] * len(y) for i, y in enumerate(y1)]); X1 = np.vstack([p["X"][:, :NF1] for p in P1 if len(p["names"])]); Y1 = np.concatenate(y1)
    oof1 = [np.zeros(len(y)) for y in y1]
    for tr, te in folds:
        mdl = mk().fit(X1[np.isin(own, tr)], Y1[np.isin(own, tr)])
        for i in te:
            if len(P1[i]["names"]): oof1[i] = mdl.predict(P1[i]["X"][:, :NF1])

    log("stage 2: GrabCut-refined candidates ...")
    P2 = [add_refined(im, p, o, topk) if len(p["names"]) else p for im, p, o in zip(images, P1, oof1)]
    y2 = [_targets(p, g) if len(p["names"]) else np.zeros(0) for p, g in zip(P2, labels)]
    own2 = np.concatenate([[i] * len(y) for i, y in enumerate(y2)]); X2 = np.vstack([p["X"] for p in P2 if len(p["names"])]); Y2 = np.concatenate(y2)
    oof2 = [np.zeros(len(y)) for y in y2]
    for tr, te in folds:
        mdl = mk().fit(X2[np.isin(own2, tr)], Y2[np.isin(own2, tr)])
        for i in te:
            if len(P2[i]["names"]): oof2[i] = mdl.predict(P2[i]["X"])

    def empty(i): return np.zeros(images[i].shape[:2], np.int32)
    grid = {t: evaluate([(labels[i], select_from_pool(P2[i], oof2[i], t) if len(P2[i]["names"]) else empty(i)) for i in range(n)]) for t in thresholds}
    thr = max(grid, key=lambda t: grid[t]["F1_at_0_5"]); log("threshold grid (dev, out-of-fold):", {t: round(v["F1_at_0_5"], 3) for t, v in grid.items()}, "-> chosen", thr)
    return dict(m1=mk().fit(X1, Y1), m2=mk().fit(X2, Y2), thr=thr, topk=topk, features=FEAT2, dev_oof=grid[thr])
