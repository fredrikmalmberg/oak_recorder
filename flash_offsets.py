"""Per-camera frame offsets from the sync flash, for the postprocess step.

The SyncFlashLED badge (worn on a lanyard, so it MOVES with the person) is triggered at take start: an RGB colour cycle,
then a white flash about 1 s later (see app.py _do_start_take / flash_control.py). Unlike detect_flash.py (fixed LED,
fixed pixel), nothing here assumes where the LED is. Per camera, over the first MAX_SECONDS only:

  * two per-pixel "onset" signals on half-res frames: blue dominance (RGB cycle) and white saturation (white flash),
  * an onset at pixel p, frame t = how much p rises vs frame t-2 AND stays up at t+1 (a rising edge, not motion),
  * pixels that keep producing onsets (animated lights, screens) are masked automatically,
  * the per-frame max over pixels is the camera's score; peaks above a robust threshold are flash candidates.

Candidates from all cameras are then voted on in time (the flash is one physical event, so cameras agree to within a
few frames). The window seen by the most cameras is the primary sync event (normally the white flash), the other one
(about 33 frames earlier) is a cross-check / fallback; each camera's second flash is chosen using its own first flash. Offsets are relative to the reference camera (cam0 when it has a
detection, else the first camera that does): offset_frames = flash_frame[cam] - flash_frame[ref], i.e. negative means the
flash appears EARLIER in that camera's file. Precision is about +-1 frame (rolling shutter / exposure start).

All limits are in seconds and converted with the take's fps (30 or 60 fps takes); the first MAX_SECONDS of each video are scanned.
A camera with no normal detection can be rescued with a lower threshold ("weak") only if it shows both flashes with the
same spacing as the other cameras. A lone event is never trusted: the take is then reported as failed.

CLI (no video written):  python flash_offsets.py <take_dir> [--max-seconds 2.5]
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

import flash_guided
from align_session import parse_log_header, parse_timestamp_log, preview_grid_layout

MAX_SECONDS = 2.5         # seconds of video scanned: the RGB cycle starts ~0.6 s in and the white flash ~1.7 s in (flash command is sent 0.2 s after take start, see app.py)
POOL = 4                 # total downscale of the score maps vs the source frame
BLUR = 5                 # blur (px, at half resolution) before pooling: roughly the LED's size
WHITE_FLOOR = 200        # white score = max(min(B,G,R) - WHITE_FLOOR, 0): only near-saturated pixels count
FEATURES = (             # name, onset level used for the hot-pixel mask, minimum peak above the noise floor
    ("blue", 20.0, 25.0),
    ("white", 12.0, 15.0),
)
HOT_S = 0.4              # a pixel with onsets in more than this many seconds worth of frames is clutter, not a flash (the LED itself only produces a few)
REF_WIDTH = 3840         # peak thresholds were tuned on 4K frames; the LED covers fewer pixels at lower resolution
# Everything below is in SECONDS and converted with the take's fps (the app records at 30 or 60 fps):
VOTE_HALF_S = 0.17       # candidates of different cameras within +-this are the same physical event (5 frames at 30 fps)
MAX_OFFSET_S = 0.5       # a larger offset between cameras is treated as a detection error (15 frames at 30 fps)
FLASH_GAP_S = (0.67, 1.5)  # between the RGB cycle and the white flash (observed ~1.1 s)
MIN_SEP_S = 0.4          # the two flash windows must be at least this far apart
TOL_S = 0.067            # two flashes "agree" / peak runs are merged within this (2 frames at 30 fps)


def frames(seconds, fps, minimum=1):
    return max(minimum, int(round(seconds * fps)))


def read_score_maps(video_path, n_frames):
    """First n_frames of a video -> {feature: float16 [T, H/POOL, W/POOL]} (None entries never occur)."""
    cap = cv2.VideoCapture(video_path)
    maps = {name: [] for name, _, _ in FEATURES}
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        half = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA).astype(np.int16)
        b, g, r = half[..., 0], half[..., 1], half[..., 2]
        blue = np.maximum(b - np.maximum(r, g), 0)  # clamped: red clothing must not produce negative-to-zero "rises"
        white = np.maximum(np.minimum(np.minimum(b, g), r) - WHITE_FLOOR, 0)
        size = (max(1, w // POOL), max(1, h // POOL))
        for name, m in (("blue", blue), ("white", white)):
            m = cv2.blur(m.astype(np.float32), (BLUR, BLUR))
            maps[name].append(cv2.resize(m, size, interpolation=cv2.INTER_AREA).astype(np.float16))
    cap.release()
    return {k: np.stack(v) if v else np.zeros((0, 1, 1), np.float16) for k, v in maps.items()}


def onset_series(maps, level, hot_n=12, return_loc=False):
    """Per-frame max over pixels of the onset signal (rise vs t-2 that is still up at t+1), clutter pixels masked.
    return_loc=True also returns the pooled (x, y) of that maximum per frame (multiply by POOL for source pixels)."""
    T = len(maps)
    series = np.zeros(T, np.float32)
    loc = np.zeros((T, 2), np.float32)
    if T < 4:
        return (series, loc) if return_loc else series

    def onset(t):
        base = maps[t - 2].astype(np.float32)
        return np.minimum(maps[t].astype(np.float32) - base, maps[t + 1].astype(np.float32) - base)

    count = np.zeros(maps.shape[1:], np.int16)
    for t in range(2, T - 1):
        count += onset(t) > level
    hot = cv2.dilate((count >= hot_n).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    for t in range(2, T - 1):
        v = onset(t)
        v[hot] = 0
        series[t] = v.max()
        if return_loc:
            y, x = np.unravel_index(np.argmax(v), v.shape)
            loc[t] = (x + 0.5, y + 0.5)
    return (series, loc) if return_loc else series


def find_peaks(series, min_peak, group_gap=2, k_mad=5.0):
    """Candidate flash frames: for every run of above-threshold frames, the frame of the strongest onset.
    Threshold = median + max(k_mad * MAD, min_peak). Returns [(frame, score)]."""
    if len(series) == 0:
        return []
    med = float(np.median(series))
    mad = float(np.median(np.abs(series - med)))
    thr = med + max(k_mad * mad, min_peak)
    idx = np.flatnonzero(series > thr)
    groups = []
    for i in idx:
        if groups and i - groups[-1][-1] <= group_gap:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return [(g[int(np.argmax(series[g]))], float(series[g].max())) for g in groups]


def best_windows(events_by_cam, half=5, n=2, min_sep=12):
    """Up to n non-overlapping time windows (centre frames) ranked by the number of cameras that have a candidate within
    +-half frames; ties -> higher summed score, then the LATER window (the white flash is more reliable than the RGB
    cycle, which can be cut off at the start of recording). Returns [{"cams": {cam: [(frame, score)]}, "center": float}]."""
    allf = [f for ev in events_by_cam.values() for f, _ in ev]
    if not allf:
        return []
    cands = []
    for c in range(min(allf), max(allf) + 1):
        members = {cam: [(f, s) for f, s in ev if abs(f - c) <= half] for cam, ev in events_by_cam.items()}
        members = {cam: m for cam, m in members.items() if m}
        if members:
            score = sum(max(s for _, s in m) for m in members.values())
            cands.append((len(members), score, c, members))
    cands.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    out = []
    for n_cams, _, c, members in cands:
        centre = float(np.median([f for m in members.values() for f, _ in m]))
        if all(abs(centre - w["center"]) >= min_sep for w in out):
            out.append({"cams": members, "center": centre})
        if len(out) == n:
            break
    return out


def pick_in_window(window, cam, expected=None):
    """The camera's candidate in the window closest to `expected` (default: the window centre); ties -> higher score."""
    m = window["cams"].get(cam)
    if not m:
        return None
    target = window["center"] if expected is None else expected
    return min(m, key=lambda fs: (abs(fs[0] - target), -fs[1]))[0]


def compute_offsets(events_by_cam, cams, fps, device_ts=None, weak_events=None):
    """events_by_cam {cam: [(frame, score)]} -> result dict (the content of flash_offsets.json, minus the traces).
    device_ts {cam: [device_ts_s per frame]} (optional) adds the timestamp-implied offset and the residual.
    weak_events: same shape as events_by_cam but found with a lower threshold; only used to rescue a camera that has no
    normal detection, and only if it shows an onset in BOTH flash windows with the same spacing as the other cameras."""
    result = {"fps": fps, "reference": None, "overall": "failed", "reason": "", "cameras": {}}
    for cam in cams:
        result["cameras"][cam] = {"flash_frame": None, "cross_check_frame": None, "offset_frames": None, "offset_ms": None,
                                  "ts_implied_offset_ms": None, "residual_ms": None, "status": "no_flash", "reason": "no flash found"}
    tol = frames(TOL_S, fps, 2)
    max_offset = frames(MAX_OFFSET_S, fps)
    gap_lo, gap_hi = FLASH_GAP_S[0] * fps, FLASH_GAP_S[1] * fps
    wins = [w for w in best_windows(events_by_cam, half=frames(VOTE_HALF_S, fps), min_sep=frames(MIN_SEP_S, fps)) if len(w["cams"]) >= 2]
    if not wins:
        result["reason"] = "no flash seen by at least 2 cameras"
        return result
    primary = wins[0]
    other = wins[1] if len(wins) > 1 else None
    if other is not None and not (gap_lo <= abs(other["center"] - primary["center"]) <= gap_hi):
        other = None
    if other is None:
        # A lone event (no RGB cycle + white flash pair with the expected spacing) could be motion or a reflection,
        # and shifting cameras by it would silently misalign the take: report it as a failure instead.
        result["reason"] = "no double flash (RGB cycle + white flash) seen by at least 2 cameras"
        return result
    # the earlier window is the RGB cycle, the later one the white flash; the sync point is whichever has more cameras
    first, second = (other, primary) if other["center"] < primary["center"] else (primary, other)

    use2 = second is not None and len(second["cams"]) >= 2
    sync_idx, other_idx = (1, 0) if use2 else (0, 1)
    sync_win = second if use2 else first
    ref = "cam0" if "cam0" in sync_win["cams"] else sorted(sync_win["cams"])[0]
    ref_f1 = pick_in_window(first, ref) if first else None

    def frames_for(cam, ref_f2):
        f1 = pick_in_window(first, cam) if first else None
        exp = None
        if None not in (f1, ref_f1, ref_f2):
            exp = ref_f2 + (f1 - ref_f1)   # this camera's own first flash predicts where its second flash must be
        f2 = pick_in_window(second, cam, exp) if second else None
        return f1, f2

    # the reference itself may have several candidates in the second window: keep the one for which the most
    # cameras' two flashes agree on the same offset (ties -> closest to the window centre)
    best = None
    for ref_f2 in ([f for f, _ in second["cams"].get(ref, [])] if second else []) or [None]:
        cand = {cam: frames_for(cam, ref_f2) for cam in cams}
        agree = sum(1 for f1, f2 in cand.values()
                    if None not in (f1, f2, ref_f1, ref_f2) and abs((f2 - ref_f2) - (f1 - ref_f1)) <= tol)
        key = (agree, -abs(ref_f2 - second["center"]) if ref_f2 is not None else 0)
        if best is None or key > best[0]:
            best = (key, cand)
    found = best[1]
    weak = set()
    if weak_events and use2 and first is not None:
        half = frames(VOTE_HALF_S, fps)
        exp_gap = second["center"] - first["center"]

        def pick(ev, win):
            c = [(f, sc) for f, sc in ev if abs(f - win["center"]) <= half]
            return min(c, key=lambda t: (abs(t[0] - win["center"]), -t[1]))[0] if c else None

        for cam in cams:
            if found[cam][1] is not None:
                continue
            ev = weak_events.get(cam, [])
            f1, f2 = pick(ev, first), pick(ev, second)
            if f1 is not None and f2 is not None and abs((f2 - f1) - exp_gap) <= tol:
                found[cam] = (f1, f2)
                weak.add(cam)
    result["reference"] = ref
    ref_f, ref_cross = found[ref][sync_idx], found[ref][other_idx]

    for cam in cams:
        r = result["cameras"][cam]
        f, cross = found[cam][sync_idx], found[cam][other_idx]
        r["flash_frame"], r["cross_check_frame"] = f, cross
        off = None
        if f is not None:
            off = f - ref_f
            r["status"], r["reason"] = "ok", ""
            if cam in weak:
                r["status"], r["reason"] = "weak", "weak flash (below the normal threshold) but both flashes found with the expected spacing"
            elif cross is not None and ref_cross is not None and abs((cross - ref_cross) - off) > tol:
                r["status"] = "check"
                r["reason"] = f"the two flashes disagree: {off:+d} vs {cross - ref_cross:+d} frames"
        elif cross is not None and ref_cross is not None:
            off = cross - ref_cross
            r["status"], r["reason"] = "single_flash", "only one of the two flashes was found"
        if off is None:
            continue
        if abs(off) > max_offset:
            r["status"], r["reason"] = "implausible", f"offset {off:+d} frames is larger than {max_offset}"
            continue
        r["offset_frames"] = int(off)
        r["offset_ms"] = off * 1000.0 / fps
        ts, ts_ref = (device_ts or {}).get(cam), (device_ts or {}).get(ref)
        fc, fr = (f, ref_f) if f is not None else (cross, ref_cross)
        if ts and ts_ref and fc < len(ts) and fr < len(ts_ref):
            r["ts_implied_offset_ms"] = (ts[fc] - ts_ref[fr]) * 1000.0
            r["residual_ms"] = r["offset_ms"] - r["ts_implied_offset_ms"]

    finalize_overall(result)
    return result


def finalize_overall(result):
    """Set overall/reason from how many cameras ended up with a usable offset."""
    cams = result["cameras"]
    n_ok = sum(1 for r in cams.values() if r["offset_frames"] is not None)
    bad = [c for c, r in cams.items() if r["offset_frames"] is None]
    result["overall"], result["reason"] = "failed", ""
    if n_ok < max(2, (len(cams) + 1) // 2):
        # a handful of shifted cameras next to a majority of unshifted ones would be worse than no alignment at all
        result["reason"] = f"flash usable in only {n_ok} of {len(cams)} cameras"
    elif bad:
        result["overall"], result["reason"] = "partial", "no usable flash in " + ", ".join(bad)
    else:
        result["overall"] = "ok"


def apply_guided(result, onsets, device_ts=None):
    """Replace the detector's offsets by the calibration-guided ones (flash_guided.guided_onsets output).
    Every camera with a guided onset is measured the same way (so a mix of two onset definitions cannot bias the
    rescued cameras by a frame); the detector's value stays in detector_offset_frames. A camera the detector had not
    accepted gets status "guided". Cameras without a guided onset (nothing visible at the projected position) keep what
    the detector gave them. Needs the reference camera to have one, otherwise nothing changes. Returns True if applied."""
    cams = result["cameras"]
    fps = result["fps"]
    ref = result["reference"]
    if not ref or onsets.get(ref, {}).get("onset") is None:
        return False
    ref_on = onsets[ref]["onset"]
    max_offset = frames(MAX_OFFSET_S, fps)
    for cam, r in cams.items():
        on = onsets.get(cam, {}).get("onset")
        r["detector_offset_frames"] = r["offset_frames"]
        r["guided_rise"] = onsets.get(cam, {}).get("rise")
        if on is None:
            if r["offset_frames"] is None:
                r["reason"] = "no flash found by the detector or at the position triangulated from the other cameras"
            continue
        off = on - ref_on
        if abs(off) > max_offset:
            continue
        was_ok = r["status"] == "ok" and r["offset_frames"] is not None
        r["flash_frame"], r["offset_frames"], r["offset_ms"] = int(on), int(off), off * 1000.0 / fps
        if not was_ok:
            r["status"] = "guided"
            r["reason"] = "flash found at the position triangulated from the other cameras (the blind detector did not accept it)"
        ts, ts_ref = (device_ts or {}).get(cam), (device_ts or {}).get(ref)
        if ts and ts_ref and on < len(ts) and ref_on < len(ts_ref):
            r["ts_implied_offset_ms"] = (ts[on] - ts_ref[ref_on]) * 1000.0
            r["residual_ms"] = r["offset_ms"] - r["ts_implied_offset_ms"]
    finalize_overall(result)
    return True


def start_frames_from(result, cams):
    """Frames to drop at the start of each camera so the flash lines up (all >= 0). Unusable cameras get 0."""
    if result["overall"] == "failed":
        return [0] * len(cams)
    offs = {c: result["cameras"].get(c, {}).get("offset_frames") for c in cams}
    usable = [o for o in offs.values() if o is not None]
    if not usable:
        return [0] * len(cams)
    lo = min(usable)
    return [(offs[c] - lo) if offs[c] is not None else 0 for c in cams]


def flash_sync_summary(result):
    """Compact view of a result for the operator UI / status file: {overall, n_ok, n_total, missing, check, reason}.
    missing = cameras without a usable offset (stay unshifted); check = offset found but the two flashes disagreed."""
    cams = result.get("cameras", {})
    missing = sorted(c for c, r in cams.items() if r.get("offset_frames") is None)
    check = sorted(c for c, r in cams.items() if r.get("offset_frames") is not None and r.get("status") in ("check", "single_flash", "weak", "guided"))
    return {"overall": result.get("overall", "failed"), "n_ok": len(cams) - len(missing), "n_total": len(cams),
            "missing": missing, "check": check, "reason": result.get("reason", "")}


def load_flash_offsets(take_dir):
    """Read the alignment saved by the postprocess step (<take_dir>/processed/flash_offsets.json) so reconstruction does not
    have to re-detect the flash. Returns None if the take has none; otherwise the saved dict plus
      "offsets":      {cam: offset_frames or None}  (flash frame in cam minus flash frame in the reference camera)
      "start_frames": {cam: frames to drop at the start of cam's video so the flash lines up (>= 0, 0 if unusable)}
    and "usable": False when overall == "failed" (then every start_frame is 0)."""
    path = os.path.join(take_dir, "processed", "flash_offsets.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    cams = list(data.get("cameras", {}))
    data["offsets"] = {c: data["cameras"][c].get("offset_frames") for c in cams}
    data["start_frames"] = dict(zip(cams, start_frames_from(data, cams)))
    data["usable"] = data.get("overall") != "failed"
    return data


def detect_flash_offsets(take_dir, max_seconds=MAX_SECONDS, return_series=False):
    """Run the detector on <take_dir>/video_cam*.mjpeg. Never raises for detection problems: they become overall='failed'."""
    series_by_cam, cams = {}, []
    try:
        paths = sorted(glob.glob(os.path.join(take_dir, "video_cam*.mjpeg")))
        if not paths:
            raise FileNotFoundError(f"no video_cam*.mjpeg in {take_dir}")
        cams = [os.path.basename(p)[len("video_"):-len(".mjpeg")] for p in paths]
        fps, width, device_ts, events, weak_events, locs = 30.0, REF_WIDTH, {}, {}, {}, {}
        for cam, path in zip(cams, paths):
            log = os.path.join(take_dir, f"frame_timestamps_{cam}.log")
            if os.path.exists(log):
                rec_w, _, log_fps, _ = parse_log_header(log)
                fps, width = log_fps or fps, rec_w or width
                device_ts[cam] = [e["device_ts_s"] for e in parse_timestamp_log(log)]
        n_frames = frames(max_seconds, fps)
        peak_scale = min(1.0, width / REF_WIDTH)   # smaller frames (e.g. 2016 px wide at 60 fps) show a weaker LED blob
        for cam, path in zip(cams, paths):
            maps = read_score_maps(path, n_frames)
            ev, series, loc = [], {}, {}
            weak_ev = []
            for name, level, min_peak in FEATURES:
                series[name], loc[name] = onset_series(maps[name], level, hot_n=frames(HOT_S, fps), return_loc=True)
                gap = frames(TOL_S, fps, 2)
                ev += find_peaks(series[name], min_peak * peak_scale, group_gap=gap)
                weak_ev += find_peaks(series[name], 0.4 * min_peak * peak_scale, group_gap=gap, k_mad=3.0)
            events[cam], weak_events[cam], series_by_cam[cam], locs[cam] = ev, weak_ev, series, loc
        result = compute_offsets(events, cams, fps, device_ts, weak_events)
        result["events"] = {c: [[int(f), round(s, 1)] for f, s in e] for c, e in events.items()}
        guided_pass(result, take_dir, cams, series_by_cam, locs, width, fps, device_ts)
    except Exception as exc:  # detection problems must never take the whole postprocess down
        result = {"fps": 30.0, "reference": None, "overall": "failed", "reason": f"{type(exc).__name__}: {exc}",
                  "cameras": {c: {"offset_frames": None, "status": "no_flash", "reason": "detection error"} for c in cams}}
    return (result, series_by_cam) if return_series else result


def flash_pixels(result, series_by_cam, locs):
    """{cam: (x, y)} source pixel of the white flash in each camera whose detection was accepted (status ok): the strongest
    onset (either score) within a few frames of its flash frame."""
    pts = {}
    for cam, r in result["cameras"].items():
        f = r.get("flash_frame")
        if r.get("status") != "ok" or f is None or cam not in locs:
            continue
        best = None
        for name, series in series_by_cam[cam].items():
            for t in range(max(f - 1, 0), min(f + 3, len(series))):
                if best is None or series[t] > best[0]:
                    best = (float(series[t]), locs[cam][name][t])
        if best is not None:
            pts[cam] = (float(best[1][0]) * POOL, float(best[1][1]) * POOL)
    return pts


def guided_pass(result, take_dir, cams, series_by_cam, locs, width, fps, device_ts):
    """If some camera was not accepted cleanly, triangulate the flash from the ones that were and look for it where it
    must be in the others (flash_guided.py). Adds result["guided"] describing what happened; never raises."""
    cam_results = result["cameras"]
    if result["overall"] == "failed" or all(r["status"] == "ok" for r in cam_results.values()):
        return
    g = result["guided"] = {"ran": False, "reason": ""}
    try:
        path, calib = flash_guided.find_calibration(take_dir, cams, width)
        if path is None:
            g["reason"] = calib
            return
        tri = flash_guided.triangulate(flash_pixels(result, series_by_cam, locs), calib, width)
        if tri is None:
            g["reason"] = "the cameras with a clean detection do not agree on one 3D position"
            return
        ref = result["reference"]
        onsets = flash_guided.guided_onsets(take_dir, cams, tri["X"], calib, fps, cam_results[ref]["flash_frame"], width)
        applied = apply_guided(result, onsets, device_ts)
        g.update({"ran": applied, "calibration": os.path.basename(path), "inliers": tri["inliers"],
                  "point_m": [round(float(v), 3) for v in tri["X"]], "reproj_px": {c: round(e, 1) for c, e in tri["errors"].items()},
                  "reason": "" if applied else "the reference camera shows no flash at the triangulated position"})
    except Exception as exc:  # the blind result stands
        g["ran"], g["reason"] = False, f"{type(exc).__name__}: {exc}"


def write_outputs(result, series_by_cam, out_dir):
    """flash_offsets.json + flash_alignment.png (score traces with the chosen frames) in out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "flash_offsets.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    if not series_by_cam:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cams = sorted(series_by_cam)
    fig, axes = plt.subplots(len(cams), 1, sharex=True, figsize=(14, 1.8 * len(cams)))
    axes = np.atleast_1d(axes)
    for ax, cam in zip(axes, cams):
        for name, _, _ in FEATURES:
            ax.plot(series_by_cam[cam][name], lw=1, label=name)
        r = result["cameras"].get(cam, {})
        for key, ls in (("flash_frame", "-"), ("cross_check_frame", ":")):
            if r.get(key) is not None:
                ax.axvline(r[key], color="k", ls=ls, lw=1)
        off = r.get("offset_frames")
        ax.set_ylabel(f"{cam}\n{'' if off is None else format(off, '+d') + ' fr'}", rotation=0, labelpad=30, va="center")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("frame (solid line = sync flash, dotted = cross-check flash)")
    fig.suptitle(f"flash sync: {result['overall']} (reference {result['reference']}) {result['reason']}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "flash_alignment.png"), dpi=80)
    plt.close(fig)


def build_overlay_png(result, cams, cell_w, cell_h, path):
    """Transparent PNG the size of the whole grid: per-cell offset labels, red banner when not fully aligned."""
    cols, rows = preview_grid_layout(len(cams))
    img = np.zeros((rows * cell_h, cols * cell_w, 4), np.uint8)
    scale = cell_w / 640.0
    banner = result["overall"] != "ok"
    bar = int(60 * scale)

    def put(text, x, y, color, size):
        for col, th in (((0, 0, 0, 255), max(2, int(5 * scale))), (color, max(1, int(2 * scale)))):
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size * scale, col, th, cv2.LINE_AA)

    for i, cam in enumerate(cams):
        row, col = divmod(i, cols)
        r = result["cameras"].get(cam, {})
        x0, y0 = col * cell_w + int(10 * scale), row * cell_h + int(34 * scale) + (bar if banner and row == 0 else 0)
        if r.get("offset_frames") is None:
            put(f"{cam}  NO FLASH - not shifted", x0, y0, (60, 60, 255, 255), 0.9)
        else:
            warn = r.get("status") in ("check", "single_flash", "weak", "guided")
            note = "  (guided)" if r.get("status") == "guided" else "  (check)" if warn else ""
            put(f"{cam}  {r['offset_frames']:+d} fr" + note, x0, y0,
                (0, 200, 255, 255) if warn else (255, 255, 255, 255), 0.9)
    if banner:
        failed = result["overall"] == "failed"
        img[:bar, :, :3] = (0, 0, 200)
        img[:bar, :, 3] = 235
        text = ("FLASH SYNC FAILED - videos NOT aligned: " if failed else "FLASH SYNC PARTIAL - ") + result["reason"]
        cv2.putText(img, text, (int(14 * scale), int(42 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 1.1 * scale,
                    (255, 255, 255, 255), max(2, int(3 * scale)), cv2.LINE_AA)
    cv2.imwrite(path, img)
    return path


def main():
    ap = argparse.ArgumentParser(description="Detect the sync flash in a take and print per-camera frame offsets.")
    ap.add_argument("take_dir")
    ap.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    ap.add_argument("--out-dir", default=None, help="also write flash_offsets.json + flash_alignment.png here")
    a = ap.parse_args()
    result, series = detect_flash_offsets(a.take_dir, a.max_seconds, return_series=True)
    print(f"overall: {result['overall']}  reference: {result['reference']}  {result['reason']}")
    print(f"{'cam':6s} {'flash':>6s} {'cross':>6s} {'offset_fr':>9s} {'ts_impl_ms':>10s} {'resid_ms':>9s}  status")
    for cam, r in result["cameras"].items():
        fmt = lambda v, p=0: "-" if v is None else f"{v:.{p}f}"  # noqa: E731
        print(f"{cam:6s} {fmt(r.get('flash_frame')):>6s} {fmt(r.get('cross_check_frame')):>6s} {fmt(r.get('offset_frames')):>9s} "
              f"{fmt(r.get('ts_implied_offset_ms'), 1):>10s} {fmt(r.get('residual_ms'), 1):>9s}  {r['status']} {r['reason']}")
    if a.out_dir:
        write_outputs(result, series, a.out_dir)


if __name__ == "__main__":
    main()
