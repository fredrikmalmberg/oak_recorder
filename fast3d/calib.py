"""Calibration for a take (fast3d step 3).

Order used by fast3d.run: a calibration given on the command line; the one
linked to the take by hand (<take>/calibration_link.json, written by the app
or fast3d.run --link); the take's own calibration when it was recorded with
one loaded (take_meta.json "calibration": {"mode": "loaded", "path": ...});
otherwise every calibration file among the search dirs whose cameras match
the take (candidate_calibrations: same labels, device ids and image size for
every camera of the take), scored with fast3d.calib_check on the take's own
2D detections, best score first (ties: nearest in time). Takes recorded
uncalibrated (calibrated after recording, like 20261022_171905/take_2) get
a calibration that way. choose_calibration below is the cheap version
without the score (take_meta's, else the matching one nearest in time).
"""
import glob
import json
import os
from datetime import datetime

DEFAULT_SEARCH_DIRS = ("output/calibration", "output/calibration_half", "/mnt/mmocap/temp_calib")


def _take_cameras(take_dir):
    with open(os.path.join(take_dir, "take_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    w, h = meta.get("resolution", (None, None))
    cams = {label: c.get("device_id") for label, c in meta.get("cameras", {}).items()}
    return meta, cams, w, h


def _calib_matches(path, cams, w, h):
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        cal = d["cameras"]
        for label, dev in cams.items():
            c = cal.get(label)
            if c is None or "extrinsics" not in c:
                return None
            intr = c["intrinsics"]
            if dev and c.get("device_id") and c["device_id"] != dev:
                return None
            if w and (intr.get("image_width"), intr.get("image_height")) != (w, h):
                return None
        return datetime.fromisoformat(d["metadata"]["timestamp"]).timestamp()
    except (OSError, KeyError, ValueError, TypeError):
        return None


LINK_FILE = "calibration_link.json"


def read_link(take_dir):
    """The calibration linked to the take by hand (app or fast3d.run --link), or None."""
    try:
        with open(os.path.join(take_dir, LINK_FILE), encoding="utf-8") as f:
            return json.load(f).get("path")
    except (OSError, ValueError):
        return None


def write_link(take_dir, path, set_by="user"):
    with open(os.path.join(take_dir, LINK_FILE), "w", encoding="utf-8") as f:
        json.dump({"path": os.path.abspath(path), "set_by": set_by, "set_unix": __import__("time").time()}, f, indent=1)


def clear_link(take_dir):
    try:
        os.remove(os.path.join(take_dir, LINK_FILE))
    except FileNotFoundError:
        pass


def candidate_calibrations(take_dir, search_dirs=DEFAULT_SEARCH_DIRS, repo_root=None):
    """All calibration files whose cameras match the take: [(path, seconds from the take start)], nearest first."""
    meta, cams, w, h = _take_cameras(take_dir)
    t_take = meta.get("start_time_unix")
    out = []
    for d in search_dirs:
        if not os.path.isabs(d) and repo_root:
            d = os.path.join(repo_root, d)
        for p in glob.glob(os.path.join(d, "*.json")):
            t = _calib_matches(p, cams, w, h)
            if t is not None:
                out.append((os.path.abspath(p), abs(t - t_take) if t_take else 0.0))
    return sorted(out, key=lambda x: x[1])


def choose_calibration(take_dir, search_dirs=DEFAULT_SEARCH_DIRS, repo_root=None):
    """Returns (path or None, how) with how one of 'take_meta', 'nearest_match (dt s)', 'none'."""
    meta, cams, w, h = _take_cameras(take_dir)
    cal = meta.get("calibration") or {}
    if cal.get("mode") == "loaded" and cal.get("path"):
        p = cal["path"]
        if not os.path.isabs(p) and repo_root:
            p = os.path.join(repo_root, p)
        if os.path.isfile(p):
            return p, "take_meta"
    t_take = meta.get("start_time_unix")
    best = None
    for d in search_dirs:
        if not os.path.isabs(d) and repo_root:
            d = os.path.join(repo_root, d)
        for p in glob.glob(os.path.join(d, "*.json")):
            t = _calib_matches(p, cams, w, h)
            if t is None:
                continue
            dt = abs(t - t_take) if t_take else 0.0
            if best is None or dt < best[0]:
                best = (dt, p)
    if best is None:
        return None, "none"
    return best[1], f"nearest_match ({best[0]:.0f} s from take start)"
