"""Dataset download, manifest and leak-free dev/eval split (4:1)."""
import glob, hashlib, os, shutil
import cv2, imagehash, numpy as np, pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

SEED = 43   # the split used for ALL reported results (seed 42 would give 127/33)
HASH_T = 8          # pHash Hamming distance <= 8  ->  same scene


def download_roboflow(api_key, out_dir, workspace="fire-bf7qe", project="egg-vzhtr", version=2):
    """Roboflow 'folder' export. NOTE: we use version 2 (160 images); the task PDF links version 1."""
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    rf.workspace(workspace).project(project).version(version).download("folder", location=out_dir)
    return out_dir


def build_manifest(raw_dir, images_dir):
    """Flatten the Roboflow export into one folder, drop exact duplicates, return a manifest."""
    os.makedirs(images_dir, exist_ok=True)
    raw = sorted(sum([glob.glob(f"{raw_dir}/**/*{e}", recursive=True) for e in (".jpg", ".jpeg", ".png")], []))
    assert raw, f"no images under {raw_dir}"
    rows = []
    for p in raw:
        parts, name = p.split("/"), os.path.basename(p)
        dst = f"{images_dir}/{name}"
        if not os.path.exists(dst):
            shutil.copy(p, dst)
        sha = hashlib.sha256(open(dst, "rb").read()).hexdigest()
        h, w = cv2.imread(dst).shape[:2]
        rows.append(dict(path=dst, name=name, stem=os.path.splitext(name)[0], h=h, w=w, sha256=sha,
                         original_split=parts[-3], roboflow_class=parts[-2]))
    return pd.DataFrame(rows).drop_duplicates("sha256").reset_index(drop=True)


def group_split(manifest, hash_t=HASH_T, test_size=0.2, seed=SEED):
    """Group near-duplicates (pHash <= hash_t OR same Roboflow parent name) and split by group."""
    paths = manifest.path.tolist(); n = len(paths)
    ph = [imagehash.phash(Image.open(p).convert("RGB")) for p in paths]
    D = np.array([[ph[i] - ph[j] for j in range(n)] for i in range(n)])
    src = [os.path.basename(p).split(".rf.")[0] for p in paths]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= hash_t or src[i] == src[j]:
                parent[find(i)] = find(j)
    m = manifest.copy()
    m["group"] = [find(i) for i in range(n)]
    dev_i, eval_i = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(m, groups=m.group))
    m["split"] = "dev"; m.loc[eval_i, "split"] = "eval"
    assert set(m[m.split == "dev"].group).isdisjoint(set(m[m.split == "eval"].group)), "LEAKAGE"
    return m
