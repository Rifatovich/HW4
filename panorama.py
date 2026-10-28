"""
ДЗ №4. Сшивка кадров в панораму (building1, building2, building3).

Алгоритм (основа - практика lab04_2):
  ORB -> BFMatcher/knnMatch + ratio test -> findHomography(RANSAC) -> warpPerspective.
Доработки для серий из многих кадров:
  * кадры предварительно проецируются на цилиндр (иначе края сильно растягиваются);
  * гомографии считаются для соседних пар и приводятся к центральному кадру;
  * кадры накладываются со взвешенным смешиванием (мягкие швы).

Запуск:  python panorama.py <папка_с_building1_2_3> [--debug] [building3=1.4]
(необязательные building<N>=<число> задают фокус цилиндра для группы)
"""
import os
import sys
import glob
import cv2 as cv
import numpy as np

MAX_SIDE = 1200      # кадры уменьшаются для скорости
N_FEATURES = 4000
RATIO = 0.80
RANSAC_PX = 3.0
MIN_INLIERS = 12
FOCAL_K = 0.9        # фокус цилиндра = FOCAL_K * ширина кадра


# ---------- загрузка ----------
def read_group(folder):
    files = []
    for pat in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        files += glob.glob(os.path.join(folder, pat))
    files = sorted(set(files))
    frames = []
    for f in files:
        im = cv.imread(f)
        if im is None:
            continue
        s = MAX_SIDE / max(im.shape[:2])
        if s < 1:
            im = cv.resize(im, None, fx=s, fy=s, interpolation=cv.INTER_AREA)
        frames.append(im)
    return frames, files


# ---------- цилиндрическая проекция ----------
def to_cylinder(img, f):
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    theta = (xs - cx) / f
    src_x = f * np.tan(theta) + cx
    src_y = (ys - cy) / np.cos(theta) + cy
    out = cv.remap(img, src_x, src_y, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    ok = ((src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)).astype(np.uint8) * 255
    ok = cv.erode(ok, np.ones((3, 3), np.uint8))
    return out, ok


# ---------- пара кадров ----------
def pair_homography(a, mask_a, b, mask_b, tag="", debug_dir=None):
    """H: точки кадра a -> точки кадра b."""
    orb = cv.ORB_create(nfeatures=N_FEATURES)
    ka, da = orb.detectAndCompute(cv.cvtColor(a, cv.COLOR_BGR2GRAY), mask_a)
    kb, db = orb.detectAndCompute(cv.cvtColor(b, cv.COLOR_BGR2GRAY), mask_b)
    if da is None or db is None:
        raise RuntimeError("не найдены ключевые точки")

    knn = cv.BFMatcher_create(cv.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    if len(good) < MIN_INLIERS:
        raise RuntimeError(f"мало совпадений ({len(good)})")

    pa = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pb = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inl = cv.findHomography(pa, pb, cv.RANSAC, RANSAC_PX)
    if H is None or inl.sum() < MIN_INLIERS:
        raise RuntimeError("гомография ненадёжна")
    n_in = int(inl.sum())
    print(f"  {tag}: keypoints {len(ka)}/{len(kb)}, good {len(good)}, inliers {n_in}")

    if debug_dir:
        sel = [m for m, k in zip(good, inl.ravel()) if k][:80]
        vis = cv.drawMatches(a, ka, b, kb, sel, None, matchColor=(0, 220, 0),
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imwrite(os.path.join(debug_dir, f"{tag}_inliers.jpg"), vis)
    return H / H[2, 2]


# ---------- приведение к одной системе координат ----------
def chain_to_center(pair_H):
    n = len(pair_H) + 1
    c = (n - 1) // 2
    T = [np.eye(3) for _ in range(n)]
    for i in range(c - 1, -1, -1):          # слева от центра
        T[i] = T[i + 1] @ pair_H[i]
    for i in range(c + 1, n):               # справа от центра
        T[i] = T[i - 1] @ np.linalg.inv(pair_H[i - 1])
    return [t / t[2, 2] for t in T], c


# ---------- сборка ----------
def compose(frames, masks, T):
    pts = []
    for im, t in zip(frames, T):
        h, w = im.shape[:2]
        c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        pts.append(cv.perspectiveTransform(c, t).reshape(-1, 2))
    pts = np.concatenate(pts)
    x0, y0 = np.floor(pts.min(0)).astype(int)
    x1, y1 = np.ceil(pts.max(0)).astype(int)
    W, H = int(x1 - x0), int(y1 - y0)
    if W <= 0 or H <= 0 or W * H > 60_000_000:
        raise RuntimeError(f"некорректный размер холста {W}x{H}")
    shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    print(f"  холст: {W} x {H}")

    acc = np.zeros((H, W, 3), np.float32)
    wsum = np.zeros((H, W), np.float32)
    for im, m, t in zip(frames, masks, T):
        M = shift @ t
        warped = cv.warpPerspective(im, M, (W, H), flags=cv.INTER_LINEAR)
        wm = cv.warpPerspective(m, M, (W, H), flags=cv.INTER_NEAREST)
        dist = cv.distanceTransform((wm > 0).astype(np.uint8), cv.DIST_L2, 3)
        dist = dist ** 2
        acc += warped.astype(np.float32) * dist[..., None]
        wsum += dist
    covered = wsum > 0
    pano = np.zeros((H, W, 3), np.uint8)
    pano[covered] = np.clip(acc[covered] / wsum[covered, None], 0, 255).astype(np.uint8)
    return pano, covered


def crop(pano, covered):
    ys, xs = np.where(covered)
    return pano[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def build_panorama(frames, name="group", debug_dir=None, focal_k=FOCAL_K):
    f = focal_k * frames[0].shape[1]
    cyl = [to_cylinder(im, f) for im in frames]
    imgs = [c[0] for c in cyl]
    masks = [c[1] for c in cyl]

    pair_H = []
    for i in range(len(imgs) - 1):
        tag = f"{name}_pair{i + 1}"
        try:
            pair_H.append(pair_homography(imgs[i], masks[i], imgs[i + 1], masks[i + 1],
                                          tag, debug_dir))
        except RuntimeError as e:
            raise RuntimeError(f"кадры {i + 1} и {i + 2}: {e}")
    T, center = chain_to_center(pair_H)
    print(f"  опорный кадр: {center + 1}")
    pano, covered = compose(imgs, masks, T)
    return crop(pano, covered)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith(("--", "building")) else "."
    debug = "--debug" in sys.argv
    focal_cfg = {}
    for a in sys.argv[1:]:
        if a.startswith("building") and "=" in a:
            k, v = a.split("=")
            focal_cfg[k] = float(v)
    dbg_dir = os.path.join(root, "debug") if debug else None
    if dbg_dir:
        os.makedirs(dbg_dir, exist_ok=True)
    for name in ("building1", "building2", "building3"):
        frames, files = read_group(os.path.join(root, name))
        print(f"{name}: {len(frames)} кадров")
        if len(frames) < 2:
            continue
        try:
            res = build_panorama(frames, name, dbg_dir, focal_cfg.get(name, FOCAL_K))
        except RuntimeError as e:
            print("  ОШИБКА:", e)
            continue
        out = os.path.join(root, f"panorama_{name}.jpg")
        cv.imwrite(out, res, [cv.IMWRITE_JPEG_QUALITY, 95])
        print(f"  сохранено: {out} ({res.shape[1]}x{res.shape[0]})")
