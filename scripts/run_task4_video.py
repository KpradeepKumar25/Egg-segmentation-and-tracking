"""Task 4: track individual eggs in a video.

  python scripts/run_task4_video.py --video task4/source.mp4 --out results/task4 [--no-guided]

Writes <out>/egg_tracking_output.mp4, egg_tracking_output_tracks.csv, identity_summary.csv, identity_events.csv.
Needs ffmpeg on the PATH. Download the video first, e.g.:  yt-dlp -f "b[height<=720][ext=mp4]" -o source.mp4 https://www.youtube.com/watch?v=Y4JyV94y2_A
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cv2
from eggcv import video

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True); ap.add_argument("--out", default="results/task4"); ap.add_argument("--no-guided", action="store_true", help="disable in-hand re-detection (v1 tracker)")
a = ap.parse_args()
fps = cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FPS)
trk, out = video.track_video(a.video, a.out, guided=not a.no_guided)
summ, events = video.audit_tracks(trk, fps)
summ.to_csv(os.path.join(a.out, "identity_summary.csv"), index=False); events.to_csv(os.path.join(a.out, "identity_events.csv"), index=False)
print(summ.round(2).to_string(index=False)); print(events.round(2).to_string(index=False)); print("video written to", out)
