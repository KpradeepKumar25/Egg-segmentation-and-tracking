"""Video egg segmentation + multi-egg tracking (Task 4).

Classical pipeline, no learned model at run time:
  1. video_seg      : LAB whiteness threshold, morphology, distance-transform watershed, shape filters
  2. run_tracker    : Hungarian association (motion-shifted mask IoU + centroid distance + area ratio), tentative-track
                      confirmation, coasting through occlusion, scene-end detection
  3. run_tracker_g  : same, plus guided re-detection of an egg that is partly hidden by the hand ("in hand")
Parameters live in the dicts V (segmenter) and G (guided re-detection) so experiments can change them in one place.
"""
import subprocess
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

LONG = 800
def prep(f):
    s = LONG / max(f.shape[:2]); return cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

V = dict(L_min=175, chroma_max=28, open_frac=0.022, area_min=0.008, area_max=0.15, solidity_min=0.88, ellipse_iou_min=0.80, aspect_max=2.0, split_peak=0.75)

def split_touching(comp, img_bgr):
    """distance-transform watershed: split one blob into several eggs if its distance map has several clear peaks"""
    m8 = comp.astype(np.uint8); dist = cv2.distanceTransform(m8, cv2.DIST_L2, 5)
    for frac in (V["split_peak"], 0.9):                                 # first a moderate, then a strict peak threshold (for heavier overlap)
        n, pk = cv2.connectedComponents((dist > frac * dist.max()).astype(np.uint8))
        if n > 2: break
    if n <= 2: return [comp]
    markers = np.zeros(comp.shape, np.int32); markers[~comp] = 1
    for k in range(1, n): markers[pk == k] = k + 1
    cv2.watershed(img_bgr, markers)
    return [(markers == k + 1) & comp for k in range(1, n) if ((markers == k + 1) & comp).sum() > 0]

def video_seg(f):
    """f: RGB frame -> label map (0 background, 1..N eggs). Pure classical: whiteness threshold, morphology, watershed, shape filters."""
    H, W = f.shape[:2]; lab = cv2.cvtColor(cv2.GaussianBlur(f, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    m = ((lab[..., 0] > V["L_min"]) & (chroma < V["chroma_max"])).astype(np.uint8)
    k = int(V["open_frac"] * W) | 1; m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = ndi.binary_fill_holes(m).astype(np.uint8); n, cc, st, _ = cv2.connectedComponentsWithStats(m, 8)
    bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR); out = np.zeros((H, W), np.int32); idx = 0
    for i in range(1, n):
        for part in split_touching(cc == i, bgr):
            a = part.sum() / part.size
            if not (V["area_min"] <= a <= V["area_max"]): continue
            cs, _ = cv2.findContours(part.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); c = max(cs, key=cv2.contourArea)
            if len(c) < 5: continue
            sol = cv2.contourArea(c) / (cv2.contourArea(cv2.convexHull(c)) + 1e-9)
            (cx, cy), (ma, mi), ang = cv2.fitEllipse(c); em = np.zeros((H, W), np.uint8); cv2.ellipse(em, ((cx, cy), (ma, mi), ang), 1, -1); em = em.astype(bool)
            eiou = (em & part).sum() / ((em | part).sum() + 1e-9)
            if sol >= V["solidity_min"] and eiou >= V["ellipse_iou_min"] and max(ma, mi) / max(min(ma, mi), 1) <= V["aspect_max"]:
                idx += 1; out[part] = idx
    return out


class Track:
    def __init__(s, mask, c, area): s.mask, s.c, s.area, s.v = mask, np.array(c, float), float(area), np.zeros(2); s.hits, s.misses, s.id, s.seen = 1, 0, None, True

def mask_shift(m, v):
    M = np.float32([[1, 0, v[0]], [0, 1, v[1]]]); return cv2.warpAffine(m.astype(np.uint8), M, (m.shape[1], m.shape[0]), flags=cv2.INTER_NEAREST) > 0

def detections(lb):
    ds_ = []
    for k in range(1, int(lb.max()) + 1):
        m = lb == k; ys, xs = np.nonzero(m)
        if len(xs): ds_.append(dict(mask=m, c=np.array([xs.mean(), ys.mean()]), area=float(m.sum())))
    return ds_

def associate(tracks, dets, big=1e6):
    """Hungarian assignment. cost = 0.4*(1-IoU with motion-shifted mask) + 0.4*centroid distance + 0.2*(1-area ratio);
    the distance gate grows the longer a track has been lost, so an egg that reappears after being hidden by the hand is re-linked."""
    if not tracks or not dets: return []
    C = np.full((len(tracks), len(dets)), big)
    for i, t in enumerate(tracks):
        pm = mask_shift(t.mask, t.v); pc = t.c + t.v; gate = (3.0 + 0.25 * min(t.misses, 20)) * np.sqrt(t.area) + 2.0 * np.linalg.norm(t.v)
        for j, d in enumerate(dets):
            iou = (pm & d["mask"]).sum() / ((pm | d["mask"]).sum() + 1e-9); dist = np.linalg.norm(pc - d["c"]); ar = min(t.area, d["area"]) / max(t.area, d["area"])
            if iou > 0 or dist < gate: C[i, j] = 0.4 * (1 - iou) + 0.4 * min(dist / gate, 1.0) + 0.2 * (1 - ar)
    r, c = linear_sum_assignment(C); return [(i, j) for i, j in zip(r, c) if C[i, j] < big / 2]

MIN_SEEN_S = 1.0
PALETTE = [(255, 80, 80), (60, 200, 255), (120, 255, 120), (255, 200, 60), (200, 120, 255), (255, 120, 200)]
def run_tracker(frame_iter, fps, writer=None, keep=None, max_age_s=8.0, confirm_hits=5, cut_thresh=40.0, scene_diff=0.5):
    tracks, next_id, rows, prev_small, ref_hist = [], 1, [], None, None; max_age = int(max_age_s * fps); trails = {}
    for fi, fr in enumerate(frame_iter):
        small = cv2.resize(fr, (64, 36)).astype(np.float32)
        if prev_small is not None and np.abs(small - prev_small).mean() > cut_thresh: tracks, trails = [], {}      # hard scene cut -> forget tracks
        prev_small = small
        dets = detections(video_seg(fr))
        hist = np.bincount((small.mean(2) / 8).astype(int).ravel().clip(0, 31), minlength=32) / small[..., 0].size            # 32-bin brightness histogram of the frame
        if dets: ref_hist = hist if ref_hist is None else 0.95 * ref_hist + 0.05 * hist                                   # what the egg scene normally looks like
        elif ref_hist is not None and 0.5 * np.abs(hist - ref_hist).sum() > scene_diff: tracks, trails, ref_hist = [], {}, None  # scene faded to a title/text slide -> stop tracking
        pairs = associate(tracks, dets); mt = {i: j for i, j in pairs}; md = {j for _, j in pairs}
        for i, t in enumerate(tracks):
            if i in mt:
                d = dets[mt[i]]
                if t.misses > int(0.5 * fps): trails.pop(t.id, None)
                t.v = 0.5 * t.v + 0.5 * (d["c"] - t.c); vmax = 0.5 * np.sqrt(t.area); t.v = t.v * min(1.0, vmax / (np.linalg.norm(t.v) + 1e-9)); t.c, t.mask, t.area = d["c"], d["mask"], 0.7 * t.area + 0.3 * d["area"]; t.hits += 1; t.misses = 0; t.seen = True
                if t.id is None and t.hits >= confirm_hits: t.id = next_id; next_id += 1
            else:
                t.misses += 1; t.seen = False; t.c = t.c + t.v; t.mask = mask_shift(t.mask, t.v); t.v *= 0.5                 # coast through occlusion (velocity capped and damped so it cannot fly off)
        tracks = [t for t in tracks if (t.id is not None and t.misses <= max_age) or (t.id is None and t.misses == 0)]
        tracks += [Track(d["mask"], d["c"], d["area"]) for j, d in enumerate(dets) if j not in md]
        vis = fr.copy()
        n_vis = 0
        for t in tracks:
            if t.id is None or (keep is not None and t.id not in keep): continue
            did = t.id if keep is None else keep[t.id]; n_vis += int(t.seen)
            rows.append(dict(frame=fi, time_s=fi / fps, id=did, cx=t.c[0], cy=t.c[1], area=t.area, detected=t.seen))
            col = PALETTE[(did - 1) % len(PALETTE)]
            if t.seen: trails.setdefault(t.id, []).append((int(t.c[0]), int(t.c[1])))
            if t.seen:
                ov = vis.copy(); ov[t.mask] = col; vis = cv2.addWeighted(ov, 0.45, vis, 0.55, 0)
                cs, _ = cv2.findContours(t.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(vis, cs, -1, col, 2); label = f"egg #{did}"
            elif t.misses <= int(0.5 * fps):
                ys, xs = np.nonzero(t.mask)
                if len(xs): cv2.rectangle(vis, (xs.min(), ys.min()), (xs.max(), ys.max()), col, 1); label = f"egg #{did} (occluded)"
                else: continue
            else: continue
            x0, y0 = int(t.c[0]) - 30, int(t.c[1]) - int(np.sqrt(t.area)) - 8
            cv2.putText(vis, label, (max(x0, 2), max(y0, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4); cv2.putText(vis, label, (max(x0, 2), max(y0, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            tr = trails.get(t.id, [])[-40:]
            for a_, b_ in zip(tr[:-1], tr[1:]): cv2.line(vis, a_, b_, col, 2)
        cv2.putText(vis, f"t={fi/fps:5.1f}s  eggs tracked: {n_vis}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4); cv2.putText(vis, f"t={fi/fps:5.1f}s  eggs tracked: {n_vis}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if writer is not None: writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    return pd.DataFrame(rows)

def frames_from_cap(cap_):
    cap_.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, f = cap_.read()
        if not ok: break
        yield prep(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))


def find_keep(pass1, fps, min_seen_s=MIN_SEEN_S):
    """choose which confirmed tracks are real eggs: seen for at least 1 s and at least half the typical egg size"""
    if len(pass1) == 0: return {}
    seen_s = pass1.groupby("id").detected.sum() / fps; first_t = pass1.groupby("id").time_s.min()
    long_ = [i for i in seen_s.index if seen_s[i] >= 3.0] or list(seen_s.index)
    area_ref = pass1.groupby("id").area.mean()[long_].median()
    good = [i for i in seen_s.index if seen_s[i] >= min_seen_s and pass1[pass1.id == i].area.mean() >= 0.5 * area_ref]
    good.sort(key=lambda i: first_t[i]); return {old: new + 1 for new, old in enumerate(good)}


G = dict(L_drop=20, chroma_add=10, open_frac=0.012, area_lo=0.30, area_hi=1.6, reach=1.2, sol_min=0.70, L_tol=30, chroma_tol=15, max_guided_s=10.0)

class GTrack(Track):
    def __init__(s, mask, c, area, lab, chroma):
        super().__init__(mask, c, area); s.area_ref = float(area); s.appL = float(lab[..., 0][mask].mean()); s.appC = float(chroma[mask].mean()); s.guided_n, s.guided = 0, False

def loose_blobs(fr, lab, chroma):
    H, W = fr.shape[:2]
    m = ((lab[..., 0] > V["L_min"] - G["L_drop"]) & (chroma < V["chroma_max"] + G["chroma_add"])).astype(np.uint8)
    k = int(G["open_frac"] * W) | 1; m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, cc, st, cen = cv2.connectedComponentsWithStats(m, 8); out = []
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a < 0.002 * H * W or a > 0.15 * H * W: continue
        mk = cc == i; cs, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); c = max(cs, key=cv2.contourArea)
        sol = cv2.contourArea(c) / (cv2.contourArea(cv2.convexHull(c)) + 1e-9)
        out.append(dict(mask=mk, c=np.array(cen[i]), area=float(a), sol=sol, L=float(lab[..., 0][mk].mean()), ch=float(chroma[mk].mean())))
    return out

def guided_pick(t, blobs, taken):
    best, ref = None, np.sqrt(t.area_ref)
    for b in blobs:
        if not (G["area_lo"] * t.area_ref <= b["area"] <= G["area_hi"] * t.area_ref): continue
        d = np.linalg.norm(b["c"] - t.c)
        if d > G["reach"] * ref or b["sol"] < G["sol_min"]: continue
        if b["L"] < t.appL - G["L_tol"] or b["ch"] > t.appC + G["chroma_tol"]: continue
        if any((b["mask"] & m).sum() / (b["mask"].sum() + 1e-9) > 0.3 for m in taken): continue
        score = d / ref + abs(b["L"] - t.appL) / G["L_tol"]
        if best is None or score < best[0]: best = (score, b)
    return None if best is None else best[1]

def run_tracker_g(frame_iter, fps, writer=None, keep=None, max_age_s=8.0, confirm_hits=5, cut_thresh=40.0, scene_diff=0.5, guided=True):
    tracks, next_id, rows, prev_small, ref_hist = [], 1, [], None, None; max_age = int(max_age_s * fps); trails = {}
    for fi, fr in enumerate(frame_iter):
        small = cv2.resize(fr, (64, 36)).astype(np.float32)
        if prev_small is not None and np.abs(small - prev_small).mean() > cut_thresh: tracks, trails = [], {}
        prev_small = small
        lab = cv2.cvtColor(cv2.GaussianBlur(fr, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32); chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
        dets = detections(video_seg(fr))
        hist = np.bincount((small.mean(2) / 8).astype(int).ravel().clip(0, 31), minlength=32) / small[..., 0].size
        if dets: ref_hist = hist if ref_hist is None else 0.95 * ref_hist + 0.05 * hist
        elif ref_hist is not None and 0.5 * np.abs(hist - ref_hist).sum() > scene_diff: tracks, trails, ref_hist = [], {}, None
        pairs = associate(tracks, dets); mt = {i: j for i, j in pairs}; md = {j for _, j in pairs}
        taken = [d["mask"] for d in dets]; blobs = None
        for i, t in enumerate(tracks):                                                    # matched tracks first (they claim their masks)
            if i in mt:
                d = dets[mt[i]]
                if t.misses > int(0.5 * fps): trails.pop(t.id, None)
                t.v = 0.5 * t.v + 0.5 * (d["c"] - t.c); vmax = 0.5 * np.sqrt(t.area); t.v = t.v * min(1.0, vmax / (np.linalg.norm(t.v) + 1e-9))
                t.c, t.mask, t.area = d["c"], d["mask"], 0.7 * t.area + 0.3 * d["area"]; t.hits += 1; t.misses = 0; t.seen = True; t.guided, t.guided_n = False, 0
                t.area_ref = 0.9 * t.area_ref + 0.1 * d["area"]; t.appL = 0.9 * t.appL + 0.1 * float(lab[..., 0][d["mask"]].mean()); t.appC = 0.9 * t.appC + 0.1 * float(chroma[d["mask"]].mean())
                if t.id is None and t.hits >= confirm_hits: t.id = next_id; next_id += 1
        for i, t in enumerate(tracks):
            if i in mt: continue
            t.guided = False; pick = None
            if guided and t.id is not None and t.guided_n < int(G["max_guided_s"] * fps):
                if blobs is None: blobs = loose_blobs(fr, lab, chroma)
                pick = guided_pick(t, blobs, taken + [o.mask for o in tracks if o is not t and o.id is not None and o.seen])
            if pick is not None:
                t.v = 0.5 * t.v + 0.5 * (pick["c"] - t.c); vmax = 0.5 * np.sqrt(t.area_ref); t.v = t.v * min(1.0, vmax / (np.linalg.norm(t.v) + 1e-9))
                t.c, t.mask = pick["c"], pick["mask"]; t.misses = 0; t.seen = True; t.guided = True; t.guided_n += 1; taken.append(pick["mask"])
            else:
                t.misses += 1; t.seen = False; t.c = t.c + t.v; t.mask = mask_shift(t.mask, t.v); t.v *= 0.5
        tracks = [t for t in tracks if (t.id is not None and t.misses <= max_age) or (t.id is None and t.misses == 0)]
        tracks += [GTrack(d["mask"], d["c"], d["area"], lab, chroma) for j, d in enumerate(dets) if j not in md]
        vis = fr.copy(); n_vis = 0
        for t in tracks:
            if t.id is None or (keep is not None and t.id not in keep): continue
            did = t.id if keep is None else keep[t.id]; n_vis += int(t.seen)
            rows.append(dict(frame=fi, time_s=fi / fps, id=did, cx=t.c[0], cy=t.c[1], area=t.area, detected=t.seen, guided=bool(t.guided)))
            col = PALETTE[(did - 1) % len(PALETTE)]
            if t.seen: trails.setdefault(t.id, []).append((int(t.c[0]), int(t.c[1])))
            if t.seen:
                ov = vis.copy(); ov[t.mask] = col; vis = cv2.addWeighted(ov, 0.30 if t.guided else 0.45, vis, 0.70 if t.guided else 0.55, 0)
                cs, _ = cv2.findContours(t.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(vis, cs, -1, col, 1 if t.guided else 2)
                label = f"egg #{did} (in hand)" if t.guided else f"egg #{did}"
            elif t.misses <= int(0.5 * fps):
                ys, xs = np.nonzero(t.mask)
                if len(xs): cv2.rectangle(vis, (xs.min(), ys.min()), (xs.max(), ys.max()), col, 1); label = f"egg #{did} (occluded)"
                else: continue
            else: continue
            x0, y0 = int(t.c[0]) - 30, int(t.c[1]) - int(np.sqrt(t.area_ref)) - 8
            cv2.putText(vis, label, (max(x0, 2), max(y0, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4); cv2.putText(vis, label, (max(x0, 2), max(y0, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            tr = trails.get(t.id, [])[-40:]
            for a_, b_ in zip(tr[:-1], tr[1:]): cv2.line(vis, a_, b_, col, 2)
        cv2.putText(vis, f"t={fi/fps:5.1f}s  eggs tracked: {n_vis}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4); cv2.putText(vis, f"t={fi/fps:5.1f}s  eggs tracked: {n_vis}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if writer is not None: writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    return pd.DataFrame(rows)



def track_video(video_path, out_dir, guided=True, max_age_s=8.0, name="egg_tracking_output"):
    """Two-pass tracking: pass 1 finds the real egg tracks, pass 2 renders only those. Returns (tracks DataFrame, mp4 path)."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path); fps = cap.get(cv2.CAP_PROP_FPS)
    ok, f0 = cap.read(); assert ok, f"cannot read {video_path}"
    Hh, Ww = prep(cv2.cvtColor(f0, cv2.COLOR_BGR2RGB)).shape[:2]
    run = (lambda it, w=None, keep=None: run_tracker_g(it, fps, w, keep, max_age_s)) if guided else (lambda it, w=None, keep=None: run_tracker(it, fps, w, keep, max_age_s))
    pass1 = run(frames_from_cap(cap)); keep = find_keep(pass1, fps)
    raw, out = os.path.join(out_dir, name + "_raw.mp4"), os.path.join(out_dir, name + ".mp4")
    wr = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (Ww, Hh))
    trk = run(frames_from_cap(cap), wr, keep); wr.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", out], check=True)
    os.remove(raw); trk.to_csv(os.path.join(out_dir, name + "_tracks.csv"), index=False)
    return trk, out


def audit_tracks(trk, fps):
    """Identity audit: coverage, gaps >= 0.5 s (re-links), implausible frame-to-frame jumps, closest approach of two eggs."""
    events, summ = [], []
    for i, g in trk.groupby("id"):
        g = g.sort_values("frame"); fr_, det, ts = g.frame.values, g.detected.values.astype(bool), g.time_s.values
        run, relinks, jumps = None, 0, 0
        for k in range(len(g)):
            if not det[k] and run is None: run = k
            if det[k] and run is not None:
                gap = (fr_[k] - fr_[run]) / fps
                if gap >= 0.5: events.append(dict(id=i, kind="re-link after gap", t_start=ts[run], t_end=ts[k], value=round(gap, 2))); relinks += 1
                run = None
        d = np.hypot(np.diff(g.cx.values), np.diff(g.cy.values)) / np.sqrt(g.area.values[1:])
        for k in np.nonzero(det[1:] & det[:-1] & (d > 0.8))[0]:
            events.append(dict(id=i, kind="position jump", t_start=ts[k], t_end=ts[k + 1], value=round(float(d[k]), 2))); jumps += 1
        summ.append(dict(id=i, first_s=ts.min(), last_s=ts.max(), frames=len(g), detected_pct=round(100 * det.mean(), 1), re_links=relinks, jumps=jumps))
    ids = sorted(trk.id.unique())
    if len(ids) >= 2:
        a = trk[trk.id == ids[0]].set_index("frame"); b = trk[trk.id == ids[1]].set_index("frame"); both = a.index.intersection(b.index)
        if len(both):
            dist = np.hypot(a.loc[both].cx - b.loc[both].cx, a.loc[both].cy - b.loc[both].cy) / np.sqrt((a.loc[both].area + b.loc[both].area) / 2)
            events.append(dict(id=0, kind="closest approach", t_start=both[int(np.argmin(dist.values))] / fps, t_end=both[int(np.argmin(dist.values))] / fps, value=round(float(dist.min()), 2)))
    return pd.DataFrame(summ), pd.DataFrame(events)
