"""Task 1: pretrained-model segmentation (OWLv2 text-prompted boxes -> SAM ViT-H masks) + recorded human review.

Pipeline per image: OWLv2 ("egg" text queries) -> boxes -> NMS -> SAM box prompt -> clean_mask -> instance NMS.
Strict thresholds first, looser fallback if nothing is found. A human then reviewed flagged images; every
manual decision is recorded below (REVIEW_* tables) so the ground truth is reproducible and disclosed.
"""
import os, cv2, numpy as np, torch
from scipy import ndimage as ndi
from torchvision.ops import nms

SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
OWL_NAME = "google/owlv2-base-patch16-ensemble"
QUERIES = ["a photo of an egg", "a white egg", "a partially visible egg cut off at the edge of the image"]
FONT = cv2.FONT_HERSHEY_SIMPLEX
PALETTE = [tuple(int(x) for x in cv2.cvtColor(np.uint8([[[(i * 47) % 180, 220, 255]]]), cv2.COLOR_HSV2RGB)[0, 0])
           for i in range(64)]

# ---------- recorded human-review decisions (indices refer to the lists built in the same order as run_task1.py) ----------
REVIEW_CHOICES = {0: [2], 1: [1], 2: [], 3: [1], 4: [1], 5: [1], 6: [], 7: [1], 8: [3], 9: [1], 10: [2], 11: [2], 12: [2], 13: [1]}  # 14 'none' images: SAM-auto candidate numbers to keep
REVIEW_REMOVE = {6: [1, 3], 14: [2], 15: [1], 21: [3], 24: [2]}     # review-list index -> egg numbers deleted
REVIEW_ADD = {5: [(610, 350)], 22: [(290, 40)]}                     # review-list index -> (x,y) click on a missed egg
REVIEW_ACCEPT = {2: [1, 2], 4: [1, 2], 7: [1], 8: [1], 10: [1], 28: [1, 2]}   # proposal-list index -> accepted extra-egg proposals
REVIEW_EXTRA_POINTS = {20: [(420, 335)], 101: [(180, 80)]}          # single-egg-list index -> (x,y) click on a missed egg
SPECIAL_BOX_FIX = {"strict_index": 20, "box": [0, 222, 506, 619], "sam_mask": 0}   # leaves/egg image: all SAM pieces kept, holes filled


# ---------- generic helpers ----------
def clean_mask(seg):
    seg = ndi.binary_fill_holes(seg)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(seg.astype(np.uint8), 8)
    if n > 2:
        seg = cc == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    return seg


def nms_instances(cands, iou_thr=0.3, contain_thr=0.6):
    kept = []
    for c in sorted(cands, key=lambda d: -d["score"]):
        ok = True
        for k in kept:
            inter = (c["seg"] & k["seg"]).sum()
            iou = inter / max((c["seg"] | k["seg"]).sum(), 1)
            contain = inter / max(min(c["seg"].sum(), k["seg"].sum()), 1)
            if iou > iou_thr or contain > contain_thr:
                ok = False; break
        if ok: kept.append(c)
    return kept


def to_label_map(kept, shape):
    lbl = np.zeros(shape, np.int32)
    for i, k in enumerate(kept, 1): lbl[k["seg"] & (lbl == 0)] = i
    return lbl


def clean_relabel(lbl):
    out = np.zeros_like(lbl); k = 0
    for i in np.unique(lbl):
        if i == 0: continue
        k += 1; out[lbl == i] = k
    return out


def label_point(seg):
    d = cv2.distanceTransform(seg.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(d), d.shape)
    return int(x), int(y)


def overlay(img_rgb, lbl, alpha=0.5, numbered=False):
    out = img_rgb.copy()
    for i in np.unique(lbl):
        if i == 0: continue
        m = lbl == i
        out[m] = (alpha * np.array(PALETTE[i % 64]) + (1 - alpha) * out[m]).astype(np.uint8)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, (255, 255, 255), 1)
        if numbered:
            x, y = label_point(m)
            cv2.putText(out, str(i), (x - 12, y + 12), FONT, 1.6, (0, 0, 0), 6)
            cv2.putText(out, str(i), (x - 12, y + 12), FONT, 1.6, (255, 255, 255), 3)
    return out


# ---------- models ----------
class Task1Models:
    def __init__(self, ckpt, device=None):
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.exists(ckpt):
            torch.hub.download_url_to_file(SAM_URL, ckpt)
        sam = sam_model_registry["vit_h"](checkpoint=ckpt).to(self.device)
        self.predictor = SamPredictor(sam)
        self.mask_gen = SamAutomaticMaskGenerator(sam, points_per_side=32, pred_iou_thresh=0.85,
                                                  stability_score_thresh=0.88, min_mask_region_area=150)
        self.owl_proc = Owlv2Processor.from_pretrained(OWL_NAME)
        self.owl = Owlv2ForObjectDetection.from_pretrained(OWL_NAME).to(self.device).eval()

    @torch.no_grad()
    def owl_boxes(self, img_rgb, queries=QUERIES):
        from PIL import Image
        H, W = img_rgb.shape[:2]
        inp = self.owl_proc(text=[queries], images=Image.fromarray(img_rgb), return_tensors="pt").to(self.device)
        out = self.owl(**inp)
        scores = torch.sigmoid(out.logits[0]).max(-1).values
        b = out.pred_boxes[0]; S = max(H, W)           # boxes are relative to the padded square image
        xyxy = torch.stack([(b[:, 0] - b[:, 2] / 2) * S, (b[:, 1] - b[:, 3] / 2) * S,
                            (b[:, 0] + b[:, 2] / 2) * S, (b[:, 1] + b[:, 3] / 2) * S], 1)
        xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, W); xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, H)
        return xyxy.float().cpu().numpy(), scores.float().cpu().numpy()

    def auto_label(self, img_rgb):
        """Automatic stage. Returns (label_map, kept, top_score, status in {strict, fallback, none})."""
        H, W = img_rgb.shape[:2]
        boxes_all, scores_all = self.owl_boxes(img_rgb)
        top = float(scores_all.max())
        self.predictor.set_image(img_rgb)

        def run(abs_thr, rel_thr, nms_thr=0.5):
            keep = scores_all >= max(abs_thr, rel_thr * top)
            b, s = boxes_all[keep], scores_all[keep]
            if len(b):
                idx = nms(torch.tensor(b), torch.tensor(s), nms_thr).numpy(); b, s = b[idx], s[idx]
            kept = []
            for bb, ss in zip(b, s):
                m, q, _ = self.predictor.predict(box=bb[None, :], multimask_output=False)
                seg = clean_mask(m[0])
                if seg.sum() < 0.3 * max((bb[2] - bb[0]) * (bb[3] - bb[1]), 1): continue   # mask barely fills its box
                kept.append(dict(seg=seg, score=float(ss) * float(q[0]), det=float(ss), box=bb))
            return nms_instances(kept, iou_thr=0.5, contain_thr=0.8)

        kept, status = run(0.10, 0.4), "strict"
        if not kept: kept, status = run(0.03, 0.5), "fallback"
        if not kept: status = "none"
        return to_label_map(kept, (H, W)), kept, top, status

    def sam_candidates(self, img_rgb, max_n=8):
        """For images where nothing was found: SAM automatic masks (human picks the egg)."""
        H, W = img_rgb.shape[:2]
        c = [dict(seg=m["segmentation"], score=m["predicted_iou"] * m["stability_score"])
             for m in self.mask_gen.generate(img_rgb) if 0.03 <= m["segmentation"].sum() / (H * W) <= 0.92]
        return nms_instances(c, iou_thr=0.6, contain_thr=0.9)[:max_n]

    def propose_extra(self, img_rgb, lbl, abs_thr=0.05, rel_thr=0.10, min_area=0.012, max_area=0.7):
        """Low-threshold OWLv2 boxes not overlapping existing eggs -> proposals a human may accept (missed eggs)."""
        H, W = img_rgb.shape[:2]
        boxes_all, scores_all = self.owl_boxes(img_rgb)
        keep = scores_all >= max(abs_thr, rel_thr * float(scores_all.max()))
        b, s = boxes_all[keep], scores_all[keep]
        if len(b) == 0: return []
        idx = nms(torch.tensor(b), torch.tensor(s), 0.5).numpy(); b, s = b[idx], s[idx]
        self.predictor.set_image(img_rgb)
        cands = []
        for bb, ss in zip(b, s):
            m, q, _ = self.predictor.predict(box=bb[None, :], multimask_output=False)
            seg = clean_mask(m[0]); a = seg.sum()
            if a < min_area * H * W or a > max_area * H * W: continue
            if (seg & (lbl > 0)).sum() / a > 0.2: continue
            cands.append(dict(seg=seg, score=float(ss), det=float(ss)))
        return nms_instances(cands, iou_thr=0.5, contain_thr=0.8)

    def seg_from_points(self, img_rgb, pos, neg=()):
        self.predictor.set_image(img_rgb)
        pts = np.array(list(pos) + list(neg)); lab = np.array([1] * len(pos) + [0] * len(neg))
        ms, sc, _ = self.predictor.predict(point_coords=pts, point_labels=lab, multimask_output=True)
        return clean_mask(ms[int(np.argmax(sc))])

    def seg_from_box_options(self, img_rgb, box):
        self.predictor.set_image(img_rgb)
        ms, sc, _ = self.predictor.predict(box=np.array(box, dtype=np.float32)[None, :], multimask_output=True)
        return [ndi.binary_fill_holes(m) for m in ms], sc
