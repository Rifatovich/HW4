"""
Подбор параметров RATIO (ratio test) и RANSAC_PX (порог RANSAC) для panorama.py.

Запуск (panorama.py должен лежать рядом с tune.py):
    python tune.py <папка_с_building1_2_3>

Для каждой комбинации параметров по всем парам соседних кадров считается:
  min inl   - число инлайеров у самой слабой пары (важнее всего: цепочка рвётся по слабому звену)
  mean %    - средняя доля инлайеров среди good matches
  err px    - средняя по парам медианная ошибка репроекции инлайеров
"""
import os
import sys
import itertools
import numpy as np
import cv2 as cv
import panorama as P

RATIOS = [0.65, 0.70, 0.75, 0.80]
THRESHOLDS = [2.0, 3.0, 4.0, 5.0]
FOCAL = {"building1": 0.9, "building2": 0.9, "building3": 1.5}   # как при итоговом запуске


def collect_pairs(frames, focal_k):
    f = focal_k * frames[0].shape[1]
    cyl = [P.to_cylinder(im, f) for im in frames]
    orb = cv.ORB_create(nfeatures=P.N_FEATURES)
    feats = []
    for im, m in cyl:
        k, d = orb.detectAndCompute(cv.cvtColor(im, cv.COLOR_BGR2GRAY), m)
        feats.append((k, d))
    bf = cv.BFMatcher_create(cv.NORM_HAMMING)
    pairs = []
    for i in range(len(feats) - 1):
        (ka, da), (kb, db) = feats[i], feats[i + 1]
        pairs.append((ka, kb, bf.knnMatch(da, db, k=2)))
    return pairs


def evaluate(pairs, ratio, thr):
    inl_counts, fracs, errs = [], [], []
    for ka, kb, knn in pairs:
        good = [p[0] for p in knn if len(p) == 2 and p[0].distance < ratio * p[1].distance]
        if len(good) < 8:
            return None
        pa = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pb = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        cv.setRNGSeed(0)
        H, mask = cv.findHomography(pa, pb, cv.RANSAC, thr)
        if H is None:
            return None
        m = mask.ravel().astype(bool)
        proj = cv.perspectiveTransform(pa, H)
        e = np.linalg.norm(proj - pb, axis=2).ravel()
        inl_counts.append(int(m.sum()))
        fracs.append(m.mean())
        errs.append(float(np.median(e[m])))
    return min(inl_counts), 100 * float(np.mean(fracs)), float(np.mean(errs))


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    for name in ("building1", "building2", "building3"):
        frames, _ = P.read_group(os.path.join(root, name))
        if len(frames) < 2:
            continue
        print(f"\n=== {name} ({len(frames)} кадров, фокус {FOCAL[name]}) ===")
        print(f"{'ratio':>6} {'RANSAC':>7} {'min inl':>8} {'mean %':>7} {'err px':>7}")
        pairs = collect_pairs(frames, FOCAL[name])
        for r, t in itertools.product(RATIOS, THRESHOLDS):
            res = evaluate(pairs, r, t)
            if res is None:
                print(f"{r:6.2f} {t:7.1f}   не удалось")
            else:
                print(f"{r:6.2f} {t:7.1f} {res[0]:8d} {res[1]:7.1f} {res[2]:7.2f}")
