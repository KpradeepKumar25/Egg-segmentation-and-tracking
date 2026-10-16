# Egg segmentation and tracking: Nosh Robotics CV Intern Task

Author: Livingstone Nikhil. Report (PDF): [`report/Egg_CV_Report.pdf`](report/Egg_CV_Report.pdf)
Video output (public Google Drive): https://drive.google.com/file/d/10tqBCNuAKUDZLcVYgcEYjmvncy57eFi1/view?usp=sharing

| Task | What | Code | Results |
|---|---|---|---|
| 1 | Segment every egg with pretrained models (OWLv2 text prompt -> SAM ViT-H), plus a documented human review | `src/eggcv/task1_sam.py`, `scripts/02_task1_sam_labels.py` | `results/task1/` |
| 2 | Leak-free 4:1 split + classical (no deep learning) segmenter + experiment log | `src/eggcv/data.py`, `src/eggcv/classical.py`, `scripts/01_make_split.py`, `scripts/03_train_classical.py` | `results/split/`, `results/task2/` (`experiment_log.md`) |
| 3 | One-shot evaluation vs Task 1 masks, quantitative + qualitative + failure analysis | `src/eggcv/metrics.py`, `scripts/04_eval_task3.py` | `results/task3/` |
| 4 | Egg tracking in the YouTube video (segmenter + tracker) | `src/eggcv/video.py`, `scripts/run_task4_video.py` | `results/task4/` |

## Headline numbers
- Dataset: Roboflow `egg-vzhtr` **version 2**, 160 images (the task PDF links `dataset/1`; version 2 is what was downloaded and used). Split 128 dev / 32 eval (4:1) with near-duplicate groups kept together (pHash Hamming <= 8 or same Roboflow parent image).
- Task 1 ground truth: 182 eggs (dev 148, eval 34), 2 images with no egg. 120 images accepted automatically as `strict`, 11 as `fallback` after review, and 29 edited by hand (12 egg picked from SAM candidates, 2 confirmed egg-free, 8 fixed, 7 missed eggs added). See report section 3.
- Task 2 (dev, 5-fold grouped out-of-fold): F1@0.5 = 0.717. Eval (frozen, one shot): F1@0.5 = 0.706 [95% CI 0.545-0.849], F1@0.75 = 0.471, foreground IoU 0.674, Dice 0.772, count MAE 0.062.
- Task 4: two eggs tracked 9.4-41.8 s. Detection coverage egg #1 81.4%, egg #2 92.0%. v2 fixes an ID swap that v1 had at about 31 s.

## Setup
Python 3.10+. Task 2, 3 and 4 run on CPU. Task 1 needs a GPU (Colab T4 is enough) and downloads ~2.6 GB of SAM weights.
```bash
git clone https://github.com/KpradeepKumar25/Egg-segmentation-and-tracking.git && cd Egg-segmentation-and-tracking
pip install -r requirements.txt
sudo apt-get install ffmpeg            # Task 4 (H.264 output); preinstalled on Colab
export ROBOFLOW_API_KEY=...            # never commit this. On Colab use Secrets (key icon) instead.
```
`notebooks/run_all_colab.ipynb` does all of the below cell by cell on Colab.

## Run
```bash
python scripts/01_make_split.py        --work work            # download + leak-free split -> work/splits/split_manifest.csv
python scripts/02_task1_sam_labels.py  --work work            # Task 1 (GPU): labels in work/task1/labels/*.npy
python scripts/03_train_classical.py   --work work            # Task 2: trains on dev only, writes work/task2/pipeline_models.joblib
python scripts/04_eval_task3.py        --work work            # Task 3: one-shot eval on the 32 eval images
yt-dlp -f "b[height<=720][ext=mp4]" -o work/source.mp4 https://www.youtube.com/watch?v=Y4JyV94y2_A
python scripts/run_task4_video.py --video work/source.mp4 --out work/task4   # Task 4 (add --no-guided for the v1 tracker)
```
The human-review step of Task 1 is stored as the `REVIEW_*` tables in `src/eggcv/task1_sam.py` (indices into lists built in a fixed order). They reproduce the submitted labels when the automatic stage gives the same output as in the original run; GPU non-determinism can shift this. `--no-review` keeps the purely automatic labels. The frozen outputs of the original run are in `results/`.

## Layout
```
src/eggcv/      data.py task1_sam.py classical.py metrics.py video.py
scripts/        01..04 + run_task4_video.py
results/        split/ task1/ task2/ (experiment_log.md, final_config.json) task3/ task4/
report/         PDF report
notebooks/      run_all_colab.ipynb
```

## Notes and known limitations
- The dev/eval split uses `GroupShuffleSplit(random_state=43)` (`SEED` in `data.py`), which gives exactly 128/32. `split_manifest.csv` in `results/split/` is the split every reported number was computed on.
- Task 1 masks are "ground truth" produced by SAM plus human review, not independent manual annotation. Task 3 measures agreement with them.
- Task 4 has no ground-truth identity labels. Identity was checked by an audit (gaps, jumps, closest approach) and by inspecting frames.
- Only 32 eval images / 34 eggs: confidence intervals are wide.

## AI tool disclosure
Claude (Anthropic) was used as a coding and analysis assistant throughout: drafting code, debugging, suggesting experiments and writing documentation. The author ran every experiment, reviewed the Task 1 masks by hand, and is responsible for the results.
