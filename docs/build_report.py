import json, pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether, ListFlowable, ListItem
from PIL import Image as PI

R = "/home/claude/repo/egg-cv-project/results"; OUT = "/home/claude/repo/egg-cv-project/report/Egg_CV_Report.pdf"
REPO = "https://github.com/KpradeepKumar25/Egg-segmentation-and-tracking"
LINK = "https://drive.google.com/file/d/10tqBCNuAKUDZLcVYgcEYjmvncy57eFi1/view?usp=sharing"
INK = colors.HexColor("#1f2937"); ACC = colors.HexColor("#111111"); SOFT = colors.HexColor("#f2f2f2"); GRID = colors.HexColor("#c8c8c8")
ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=9.4, leading=13.6, textColor=INK, spaceAfter=6, alignment=TA_JUSTIFY)
h1 = ParagraphStyle("h1", parent=body, alignment=TA_LEFT, fontName="Helvetica-Bold", fontSize=14.5, leading=18, textColor=ACC, spaceBefore=14, spaceAfter=6, keepWithNext=1)
h2 = ParagraphStyle("h2", parent=body, alignment=TA_LEFT, fontName="Helvetica-Bold", fontSize=10.8, leading=14, spaceBefore=6, spaceAfter=3)
title = ParagraphStyle("t", parent=body, alignment=TA_LEFT, fontName="Helvetica-Bold", fontSize=22, leading=27, textColor=ACC)
cap = ParagraphStyle("cap", parent=body, alignment=TA_LEFT, fontSize=8, leading=10.5, textColor=colors.HexColor("#4b5563"))
cell = ParagraphStyle("c", parent=body, alignment=TA_LEFT, fontSize=7.8, leading=10, spaceAfter=0)
cellb = ParagraphStyle("cb", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
call = ParagraphStyle("call", parent=body, alignment=TA_LEFT, backColor=SOFT, borderPadding=(6, 8, 6, 8), leftIndent=8, rightIndent=8, spaceBefore=6, spaceAfter=10)
P = lambda t, s=body: Paragraph(t, s)
def bl(items): return ListFlowable([__import__("reportlab.platypus", fromlist=["ListItem"]).ListItem(P(i), leftIndent=12) for i in items], bulletType="bullet", start="-", leftIndent=12, bulletFontSize=8)
def tbl(rows, w):
    d = [[Paragraph(str(c), cellb if r == 0 else cell) for c in row] for r, row in enumerate(rows)]
    t = Table(d, colWidths=[x * mm for x in w], repeatRows=1)
    st = [("GRID", (0, 0), (-1, -1), .4, GRID), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 0), (-1, 0), ACC), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LEFTPADDING", (0, 0), (-1, -1), 2.5), ("RIGHTPADDING", (0, 0), (-1, -1), 2.5)]
    st += [("BACKGROUND", (0, r), (-1, r), SOFT) for r in range(2, len(rows), 2)]; t.setStyle(TableStyle(st)); t.hAlign = 'CENTER'; t.spaceBefore = 2; t.spaceAfter = 9; return t
def fig(path, caption, width=174):
    w, h = PI.open(path).size; im = Image(path, width * mm, width * mm * h / w); im.hAlign = 'CENTER'; return KeepTogether([im, P(caption, cap), Spacer(1, 6)])
def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 7.6); c.setFillColor(colors.HexColor("#6b7280"))
    c.drawString(18 * mm, 10 * mm, "Egg segmentation and tracking - K Pradeep Kumar - Nosh Robotics CV Intern Task"); c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {d.page}"); c.restoreState()

ev = json.load(open(f"{R}/task3/eval_metrics.json")); E = ev["eval"]; CI = ev["ci"]; log = pd.read_csv(f"{R}/task2/experiment_log.csv")
s = []
s += [P("Egg segmentation and tracking", title), Table([[P("<b>Task</b>", body), P("Nosh Robotics CV Intern Task", body)], [P("<b>Author</b>", body), P("K Pradeep Kumar", body)], [P("<b>Repository</b>", body), P(f"<a href='{REPO}' color='#1a1a1a'>{REPO}</a>", body)]], colWidths=[22*mm, 148*mm], style=TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("LINEBELOW",(0,0),(-1,-1),0.4,GRID),("LINEABOVE",(0,0),(-1,0),0.4,GRID),("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),0)]), hAlign="LEFT"), Spacer(1, 4)]
s += [P(f"<b>Summary.</b> (1) I made reference masks for all 160 images with OWLv2 + SAM ViT-H and reviewed them by hand. (2) I built a classical segmenter (no deep learning at inference) on a leak-free development split of 128 images; my experiment log covers six trials including the failures. (3) On the 32 held-out images it reaches F1@0.5 = {E['F1_at_0_5']:.3f} (95% CI {CI['F1_at_0_5'][0]:.2f}-{CI['F1_at_0_5'][1]:.2f}) against the Task 1 masks. (4) I adapted the same ideas to the video, where two eggs are tracked with stable IDs through occlusion and hand movements. "
        f"<b>Video output:</b> <a href='{LINK}' color='#1a1a1a'>{LINK}</a>", call)]
s += [tbl([["Task", "What was delivered", "Headline result"],
           ["1", "SAM-based masks for 160 images (182 eggs), 29 images edited by hand", "120 images accepted automatically; 2 images contain no egg"],
           ["2", "Leak-free 4:1 split, classical segmenter, experiment log T0-T5", "Dev (5-fold out-of-fold) F1@0.5 0.717"],
           ["3", "One-shot evaluation, CIs, failure taxonomy, qualitative figures", f"Eval F1@0.5 {E['F1_at_0_5']:.3f}, fg IoU {E['fg_IoU']:.3f}, count MAE {E['count_MAE']:.3f}"],
           ["4", "Video segmentation + tracker, public output video", "2 eggs, IDs consistent after fixing a v1 swap; coverage 81% / 92%"]], [12, 94, 68])]

s += [P("1. Dataset and leak-free split", h1),
      P("<b>Data.</b> Roboflow project <i>egg-vzhtr</i>, <b>version 2</b>, folder export: 160 images, all 640 x 640 (158 labelled 'egg', 2 'Unlabeled'). The task PDF links dataset version 1; version 2 is what was downloaded and used throughout, and the exact download call is in <font face='Courier'>src/eggcv/data.py</font>. I ignored Roboflow's own train/valid/test split (112/32/16). No exact byte-duplicate files were found."),
      P("<b>Leakage control.</b> Images of the same scene would inflate the evaluation, so I grouped images before splitting: two images are in one group if their perceptual hashes (pHash, 64-bit) differ by at most 8 bits, or if they share a Roboflow parent filename (the part before <font face='Courier'>.rf.</font>), merged with union-find. This gave 157 groups (3 contain more than one image, largest 2), so this export contains little near-duplication. I then split the groups with <font face='Courier'>GroupShuffleSplit(test_size=0.2, random_state=43)</font> into <b>128 dev / 32 eval images</b> (exactly 4:1), and the code asserts that no group appears on both sides. Everything in Task 2 (feature design, model fitting, threshold choice, every trial in the log) used the dev split only; the eval split was touched once, in Task 3, after the final model was frozen with a SHA-256 hash of the model file.")]

s += [P("2. Task 1: SAM / pretrained-model segmentation", h1),
      P("<b>Method.</b> SAM alone has no notion of 'egg'. My first attempt (SAM automatic masks, filtered by ellipse fit, solidity and skin colour) was not reliable enough to serve as a reference, so I discarded it. The final method prompts SAM with detections: OWLv2 (<font face='Courier'>owlv2-base-patch16-ensemble</font>) is queried with three phrases ('a photo of an egg', 'a white egg', 'a partially visible egg cut off at the edge of the image'); boxes are kept if their score is at least max(0.10, 0.4 x best score in the image) and merged with NMS (IoU 0.5); each box prompts SAM ViT-H (single mask); holes are filled and only the largest component kept; masks that fill less than 30% of their box are rejected; a final instance NMS removes duplicates. If nothing passes, a looser fallback (0.03 / 0.5) is tried."),
      P("<b>Human review (disclosure).</b> I checked all 160 results visually: every fallback image and every image with two or more eggs, a random sample of 12 single-egg images, and a full contact sheet of all single-egg images. I then made these edits, all recorded in <font face='Courier'>task1_sam.py</font>: for 14 images where nothing was found, picking the correct SAM candidate (12) or confirming that the image has no egg (2); deleting wrong eggs or adding a missed egg by one click (8 images); accepting extra eggs proposed by a low-threshold OWLv2 pass or by a click (7 images); and one image where leaves cut the egg, where all SAM pieces were kept. Final status counts: strict 120, fallback 11, manual pick 12, manual fix 8, manual add 7, no egg 2. Total <b>182 eggs</b> (dev 148, eval 34); 17 images have two or more eggs."),
      P("<b>Caveat.</b> These masks are SAM output corrected by me, not independent manual annotation. Task 3 therefore measures agreement with this reference, not absolute accuracy.")]

s += [P("3. Task 2: classical segmenter and experiment log", h1),
      P("<b>Final method (trial T5).</b> Propose-and-rank, using only classical image operations plus a small Random Forest: (a) compute five channels (L, a, b, saturation, and 'whiteness' = L - 1.5 x chroma), threshold each with Otsu in <i>both polarities</i>, clean with open+close (9x9 ellipse), fill holes, drop regions under 2% of the image: (b) Describe each candidate with 25 shape, edge, contrast and colour features (area, solidity, ellipse fit, border contact, edge strength, contrast, texture, colour spread, centre distance, mean L, chroma, circularity, how many other candidates agree with it, which channel produced it). (c) A Random Forest (300 trees) predicts each candidate's IoU with the reference egg. (d) The top six candidates are refined with GrabCut (4 iterations at 320 px), added to the pool, and a second Random Forest re-ranks raw plus refined candidates. (e) The best candidate is always kept; others are kept if the predicted IoU is at least 0.5 and they do not overlap a kept candidate (NMS IoU 0.3). Dev scores are out-of-fold (5-fold GroupKFold on the pHash groups) so no image is scored by a model that saw it."),
      P("<b>Experiment log.</b> Each row gives what changed, why, the parameters, the dev results and the decision. Failures are kept on purpose; the log also lives in <font face='Courier'>results/task2/experiment_log.md</font>.")]
md = open(f"{R}/task2/experiment_log.md").read().strip().splitlines()[2:]
rows = [["Trial", "What changed / why", "Parameters", "P", "R", "F1", "mIoU", "fgIoU", "MAE", "Decision"]]
for ln in md:
    c = [x.strip() for x in ln.strip("|").split("|")]
    rows.append([c[0], f"{c[1]}. <i>{c[2]}</i>", c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10].replace("**", "").replace(".", ". ", 1) if False else c[10].replace("**", "")])
s += [tbl(rows, [10, 31, 31, 12, 12, 12, 12, 12, 12, 30])]
s += [P("<font size=7.5>m-IoU: mean IoU of matched predicted/reference eggs. fg IoU: foreground (all-egg) IoU per image, averaged. P, R and F1 count a predicted egg as correct when it matches a reference egg at IoU >= 0.5 (Hungarian matching).</font>"),
      P("<b>What the trials taught.</b> (T0, T1) Assuming that the egg is the bright region fails because many eggs are darker than their background (cloth, trays, hands): F1 0.36-0.41. (T2) Using both polarities and five channels fixed the count (MAE 1.5 to 0.46), but my hand-set score weights picked small regions. (T3) Letting the data choose the weights (Ridge) lifted F1 from 0.49 to 0.68. (T4) A non-linear ranker gave 0.715, and an oracle check showed the candidate pool, not the ranking, was the ceiling: only 45% of eggs had a candidate at IoU >= 0.8. (T5) GrabCut-refined candidates raised that share to 53% and matched-egg IoU from 0.80 to 0.81, with F1 roughly unchanged (0.715 to 0.717). T5 was frozen as final. My honest reading: the last step improved boundary quality, not detection.")]

s += [P("4. Task 3: evaluation against the Task 1 masks", h1),
      P("I ran the frozen model (hash-checked) once on the 32 eval images (34 reference eggs). Predicted instances are matched to reference eggs by the Hungarian algorithm on IoU."),
      tbl([["Metric (eval, frozen model)", "Value", "95% bootstrap CI", "Dev out-of-fold"],
           ["Precision / recall @ IoU 0.5", f"{E['precision']:.3f} / {E['recall']:.3f}", "", "0.724 / 0.710"],
           ["F1 @ IoU 0.5", f"{E['F1_at_0_5']:.3f}", f"{CI['F1_at_0_5'][0]:.3f} - {CI['F1_at_0_5'][1]:.3f}", "0.717"],
           ["F1 @ IoU 0.75", f"{E['F1_at_0_75']:.3f}", "", ""],
           ["Foreground IoU / Dice (per image)", f"{E['fg_IoU']:.3f} / {E['fg_Dice']:.3f}", f"IoU {CI['fg_IoU'][0]:.3f} - {CI['fg_IoU'][1]:.3f}; Dice {CI['fg_Dice'][0]:.3f} - {CI['fg_Dice'][1]:.3f}", "0.651 / 0.745"],
           ["Egg-count MAE / exact count", f"{E['count_MAE']:.3f} / {E['exact_count_acc']*100:.1f}%", "", "0.102"],
           ["False-positive regions", f"{E['false_positive_regions']}", "", ""],
           ["Mean IoU of matched eggs", "0.810", "", "0.813"]], [58, 32, 52, 32]),
      Spacer(1, 6),
      P("<b>Reading the numbers.</b> Counting works well (the number of eggs is right in 93.8% of images) and found eggs are outlined well (matched IoU 0.81). The weakness is <i>which region</i> is called the egg and how far it leaks: F1 drops from 0.71 to 0.47 when the IoU bar moves from 0.5 to 0.75. Eval and dev agree within the confidence interval (0.706 vs 0.717), so there is no sign of overfitting to dev. With only 32 images the interval is wide (about +/- 0.15), so differences of a few points are not meaningful."),
      P("<b>Failure taxonomy</b> (per reference egg, 34 total): good (IoU >= 0.75): 16; boundary error (0.5-0.75): 8; leak, where the region is far larger than the egg: 6; mislocalised: 3; merge of two eggs: 1. Plus 4 false-positive regions. Dominant failure: leak, where a white egg touches a similarly pale wall, bowl, cloth or hand and thresholding cannot separate them (see figure). Mislocalised cases are images where a bright tray or cup edge scored higher than the egg. A pure colour or edge cue has no notion of 'egg shape', which a deep model supplies."),
      fig(f"{R}/task3/analysis_plots.png", "Figure 1. Eval split: per-image foreground IoU, IoU against egg size, failure taxonomy and count agreement."),
      fig(f"{R}/task3/worst_cases.png", "Figure 2. The eight worst eval images. Green = Task 1 mask, red = classical prediction. Mostly leaks into similarly pale surroundings, plus two mislocalised cases."),
      fig(f"{R}/task3/best_cases.png", "Figure 3. The four best eval images: isolated eggs with clear contrast to the background.", 150)]

s += [P("5. Task 4: tracking individual eggs in the video", h1),
      P(f"<b>Video:</b> https://www.youtube.com/watch?v=Y4JyV94y2_A (1440 frames at 24 fps). <b>Output (public link):</b> <a href='{LINK}' color='#1a1a1a'>{LINK}</a>. Two eggs are on screen from 9.4 s to 41.8 s; each box and mask carries a stable ID (#1, #2) and a trail."),
      P("<b>Segmenter (video-specific).</b> Re-running the Task 2 pipeline per frame was too slow and, with a single scene, unnecessary, so I reduced the segmenter to its essentials: work at 800 px; whiteness in LAB (L above 175, chroma below 28); morphology; distance-transform watershed to split touching eggs; shape filters on solidity, ellipse fit and minimum area (0.8% of the frame). <b>Check against SAM:</b> on 30 sampled frames (50 egg instances, 11-40 s) the masks agree with SAM (ViT-B, box prompts) with mean IoU 0.949 (median 0.962, minimum 0.819; every instance above 0.75). This tests boundary quality only, not missed eggs."),
      P("<b>Tracker.</b> Detections are associated to tracks with the Hungarian algorithm on a cost that combines IoU of the motion-shifted mask, centroid distance and area ratio. New objects must be seen for five frames before they become tracks. When an egg is hidden, the track coasts on a damped, capped velocity for up to 8 s with a growing gate. After the first pass, ghost tracks (seen under 1 s, or much smaller than a typical egg) are dropped and a second pass renders only the real tracks. Scene end is detected from the brightness histogram."),
      P("<b>An error found by auditing, and the v2 fix.</b> My first tracker (v1) looked fine in spot checks, but an identity audit (gaps over 0.5 s, jumps larger than 0.8 egg-sizes per frame, closest approach) and a frame-by-frame look at 28-34 s showed that v1 <b>swapped the two IDs at about 31 s</b>: both eggs were lost behind the hand, and the egg that reappeared was nearest to #2's last position. In version 2 I added <i>guided re-detection</i>: when a track has no clean detection, loose white blobs near its last position are matched on size, brightness, colour and solidity, so the track follows an egg while it is held in the hand and is not free to be claimed by the wrong ID. Frames produced this way are labelled '(in hand)' in the video and flagged in the tracks file. v2 is the submitted video."),
      tbl([["Per-egg result (v2)", "Egg #1", "Egg #2"],
           ["Track frames (9.4-41.8 s)", "778", "778"],
           ["Detection coverage v1 -> v2", "69.5% -> 81.4%", "97.4% -> 92.0%"],
           ["...of which in-hand re-detection frames", "27", "23"],
           ["Longest stretch without detection (v2)", "4.12 s (fully hidden)", "2.08 s"],
           ["Identity swaps", "none seen in v2 (v1: swapped at ~31 s)", "none seen in v2"]], [66, 54, 54]),
      P("Egg #2's coverage <i>drops</i> in v2 because in v1 its track had jumped to the other egg, which was cleanly detected; v2 follows the real egg #2 through the hand, so the lower number is the more correct one."),
      P("<b>Robustness.</b> One parameter changed at a time on the 24-38 s window: whiteness L threshold, chroma limit, solidity, minimum area and the ellipse-fit limit at 0.88 all leave the tracks unchanged (position shift 0.0-0.2 px, still 2 tracks). The exception is the ellipse-fit limit loosened to 0.72, which creates a third track and drops egg #1's detections from 312 to 223 frames; the default (0.80) is therefore close to a fragile edge, and this is a known weakness."),
      fig(f"{R}/task4/identity_audit.png", "Figure 4. Identity audit (v1): detection timeline and re-links. The re-link of egg #2 at 30.4-31.2 s is where the swap happened."),
      fig(f"{R}/task4/swap_check_v1_vs_v2.png", "Figure 5. Side-by-side of v1 and v2 for 28.5-34 s. At 31.5 s, v2 labels the egg on the left #1 (correct, it was put down there); v1 labels it #2."),
      fig(f"{R}/task4/hand_check_v2.png", "Figure 6. v2 following the eggs while held in the hand ('in hand' frames).", 150),
      fig(f"{R}/task4/video_vs_sam.png", "Figure 7. Video segmenter masks against SAM on sampled frames (mean IoU 0.949).", 150),
      fig(f"{R}/task4/trajectories.png", "Figure 8. Egg positions over time in v2; black dots mark in-hand re-detection. Both IDs stay on continuous paths.", 150)]

s += [P("6. Limitations", h1),
      bl(["No ground-truth identity labels exist for the video; identity was checked by an audit and by inspecting frames, not by a numeric ID-switch metric. Absence of swaps is therefore 'none found', not proof.",
          "Egg #1 is not detected for up to 4.1 s while completely hidden by the hand; its position then comes from the motion model and the in-hand search.",
          "The video segmenter is tuned to this scene (pale eggs, one lighting). It is not a general egg detector; the Task 2 segmenter is the general one.",
          "Evaluation uses 32 images and 34 eggs: wide confidence intervals. The reference is SAM plus my own review, not independent annotation.",
          "Classical segmentation leaks into pale surroundings (6 of 34 eggs). A small learned model would likely fix most of these, but Task 2 restricts the method to classical vision.",
          "The dev/eval split is one random draw (seed 43); a different seed would give different eval images and slightly different scores."])]

s += [P("7. Conclusion", h1),
      P("I took the project from raw photos to tracked eggs in a video in four steps. Every reported number comes from a dev/eval split that I fixed before the held-out images were used."),
      bl(["<b>Task 1:</b> prompting SAM with OWLv2 text detections gave usable masks for all 160 images (182 eggs). About 18% of the images (29) needed a manual fix, and each fix is recorded in the repository.",
          "<b>Task 2:</b> a classical propose-and-rank segmenter reached F1@0.5 = 0.717 on dev. Learning the ranking, not the thresholding, gave the biggest gain (F1 0.49 to 0.68). GrabCut refinement improved boundary quality (matched IoU 0.80 to 0.81) but not detection.",
          f"<b>Task 3:</b> on the 32 held-out images the method scored F1@0.5 = {E['F1_at_0_5']:.3f} (95% CI {CI['F1_at_0_5'][0]:.2f}-{CI['F1_at_0_5'][1]:.2f}), with the egg count correct in {E['exact_count_acc']*100:.1f}% of images. It is good at finding eggs and weaker at outlining them tightly (F1@0.75 = {E['F1_at_0_75']:.3f}); the main failure is leaking into pale surroundings.",
          "<b>Task 4:</b> two eggs are tracked through occlusion and hand movement with stable IDs. My audit exposed an ID swap in the first tracker, and guided 'in hand' re-detection in v2 fixed it."]),
      P("<b>What I would do next:</b> add a small learned model for the boundary, since colour and edge cues cannot tell a pale egg from a pale wall; hand-label identities in a few video segments to measure ID switches numerically; and test on more videos than this one scene.")]

s += [P("8. AI tool disclosure", h1),
      P("I used Claude (Anthropic) as an assistant for writing and debugging code, suggesting experiments (for example the polarity-agnostic candidates and the in-hand re-detection), checking results and drafting this report. I ran the experiments in Colab, reviewed the Task 1 masks by hand, made the keep/discard decisions, and can explain each method. Claude made one error along the way (it declared v1 free of ID swaps before the side-by-side check); my audit caught it and it is reported in section 5.")]

s += [P("9. Reproducing the results", h1),
      Paragraph(f"The code is in the public GitHub repository <a href='{REPO}' color='#1a1a1a'>{REPO}</a>. Repository layout, setup and run commands are in its README. In short: <font face='Courier'>pip install -r requirements.txt</font>, set <font face='Courier'>ROBOFLOW_API_KEY</font> as an environment variable (never committed), then run <font face='Courier'>scripts/01_make_split.py</font>, <font face='Courier'>02_task1_sam_labels.py</font> (GPU), <font face='Courier'>03_train_classical.py</font>, <font face='Courier'>04_eval_task3.py</font> and <font face='Courier'>run_task4_video.py</font>. A Colab version is in <font face='Courier'>notebooks/run_all_colab.ipynb</font>. Frozen outputs of the submitted run are in <font face='Courier'>results/</font>.", ParagraphStyle("left8", parent=body, alignment=TA_LEFT))]
SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=16*mm, bottomMargin=17*mm, title="Egg segmentation and tracking - report", author="K Pradeep Kumar").build(s, onFirstPage=footer, onLaterPages=footer)
print("built")
