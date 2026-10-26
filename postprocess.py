"""Background per-take postprocessing worker for app.py's capture mode.

A single daemon thread drains a queue of finished take directories and runs
PROCESSING_STEPS over each one in order. Deliberately a plain Python
worker (threading.Thread + queue.Queue calling ordinary functions), not an
AI-agent-spawning mechanism -- see README.md's "Capture mode" section. New
capabilities get added here over time as more functions appended to
PROCESSING_STEPS, each taking (take_dir, meta) and reading/writing under
take_dir.
"""
import json
import os
import queue
import threading

import flash_offsets
from align_session import (
    convert_mjpegs_to_grid_mp4,
    grid_cell_size,
    parse_log_header,
    preview_grid_layout,
)


def _step_aligned_grid_video(take_dir, meta):
    """Flash-aligned grid video (replaces the old index-tiled one at the same path).

    Detects the sync flash in every camera (flash_offsets.py), writes flash_offsets.json /
    flash_alignment.png next to the video, and tiles the cameras shifted by their flash offsets.
    If the flash isn't found (or only in some cameras) the grid is still built -- unshifted for the
    affected cameras -- and the problem is burned into the video, so a take is never silently
    "aligned" when it isn't.
    """
    processed_dir = os.path.join(take_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    labels = list(meta["cameras"])
    cameras = list(meta["cameras"].values())
    video_paths = [cam["video_path"] for cam in cameras]
    # Resolution isn't in take_meta.json itself -- read it from any one
    # camera's own log header (all cameras in a take share cfg["camera"]'s
    # record_width/height) so the grid's cells get the SOURCE frames' real
    # aspect ratio instead of being distorted into a fixed 16:9 slice (see
    # grid_cell_size's docstring -- matters for the half-res native-sensor
    # profile, whose full-sensor readout isn't 16:9 like 4K's ISP crop).
    frame_w, frame_h, _fps, _offset = parse_log_header(cameras[0]["log_path"])

    result, series = flash_offsets.detect_flash_offsets(take_dir, return_series=True)
    flash_offsets.write_outputs(result, series, processed_dir)
    print(f"[Postprocess] Flash sync for {take_dir}: {result['overall']} {result['reason']}")
    cols, _rows = preview_grid_layout(len(labels))
    cell_w, cell_h = grid_cell_size(cols, frame_w, frame_h)
    overlay_png = flash_offsets.build_overlay_png(
        result, labels, cell_w, cell_h, os.path.join(processed_dir, "flash_overlay.png"))
    convert_mjpegs_to_grid_mp4(
        video_paths, meta["fps"], processed_dir, frame_w, frame_h,
        start_frames=flash_offsets.start_frames_from(result, labels), overlay_png=overlay_png)
    return {"flash_sync": flash_offsets.flash_sync_summary(result)}


PROCESSING_STEPS = [_step_aligned_grid_video]


def process_take(take_dir):
    """Runs every step; a step may return a dict, which is merged into the take's status file (e.g. {"flash_sync": ...})."""
    with open(os.path.join(take_dir, "take_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    extra = {}
    for step in PROCESSING_STEPS:
        extra.update(step(take_dir, meta) or {})
    return extra


STATUS_FILENAME = "postprocess_status.json"


def read_status(take_dir):
    """Read a take's postprocess status, written by PostprocessWorker below.
    Returns "queued" if the file doesn't exist yet (job enqueued but the
    worker hasn't picked it up), or the last status the worker wrote
    ("processing", "done", "failed") plus "error" when failed. This is the
    UI's only source of truth for postprocess progress -- deliberately not
    inferred from whether the output mp4 exists, since that can't tell
    "still processing" apart from "failed".
    """
    path = os.path.join(take_dir, STATUS_FILENAME)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"status": "queued"}


def flash_sync_text(status):
    """(text, is_problem) for the flash-sync part of a take's status dict; ("", False) when the take has no flash info
    (e.g. postprocessed before this existed, or still running)."""
    fs = status.get("flash_sync")
    if not fs:
        return "", False
    if fs["overall"] == "ok":
        note = f" (check {', '.join(fs['check'])})" if fs.get("check") else ""
        return f"flash sync OK, {fs['n_ok']}/{fs['n_total']} cameras{note}", bool(fs.get("check"))
    if fs["overall"] == "partial":
        return f"FLASH SYNC PARTIAL: no flash in {', '.join(fs['missing'])}", True
    return f"FLASH SYNC FAILED, not aligned: {fs['reason']}", True


def status_text(status):
    """One-line operator text for a take's postprocess status, including the flash-sync outcome."""
    base = {
        "queued": "queued for postprocessing",
        "processing": "processing...",
        "done": "postprocessed",
        "failed": f"postprocessing failed ({status.get('error', 'unknown error')})",
    }.get(status["status"], status["status"])
    text, problem = flash_sync_text(status)
    if not text:
        return base
    return f"{base} -- " + (f"⚠️ **{text}**" if problem else text)


def _write_status(take_dir, status, error=None, extra=None):
    payload = dict(extra or {})
    payload["status"] = status
    if error is not None:
        payload["error"] = str(error)
    path = os.path.join(take_dir, STATUS_FILENAME)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)  # atomic -- readers never see a half-written file


class PostprocessWorker(threading.Thread):
    """One job at a time, serialized -- so postprocessing never competes with
    itself for CPU/ffmpeg, but a new take's recording is never blocked by a
    previous take's still-running postprocessing (see app.py's capture loop:
    jobs are only ever enqueued, never waited on before starting the next take).

    Progress is reported back via a small JSON status file per take (see
    read_status/_write_status above) rather than in-memory state, so it
    survives an app.py restart and can be read from any process.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.queue = queue.Queue()

    def run(self):
        while True:
            take_dir = self.queue.get()
            if take_dir is None:
                return
            _write_status(take_dir, "processing")
            try:
                extra = process_take(take_dir)
            except Exception as exc:
                print(f"[Postprocess] {take_dir} failed: {exc}")
                _write_status(take_dir, "failed", error=exc)
            else:
                _write_status(take_dir, "done", extra=extra)
                text, problem = flash_sync_text(extra)
                if problem:
                    print(f"[Postprocess] WARNING {take_dir}: {text}")
            self.queue.task_done()


# ---------------------------------------------------------------------------
# fast3d: CPU-only MediaPipe 3D skeleton + face (fast3d/run.py), its own worker
# ---------------------------------------------------------------------------

FAST3D_STATUS = os.path.join("processed", "fast3d_status.json")
REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def read_fast3d_status(take_dir):
    """fast3d/run.py's status file (status, stage, progress, error, calibration{path, how, verdict, median_px,
    flagged}), or {"status": "none"} for a take fast3d never ran on."""
    try:
        with open(os.path.join(take_dir, FAST3D_STATUS), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"status": "none"}


def fast3d_npz_path(take_dir):
    return os.path.join(take_dir, "processed", "fast3d", "fast3d.npz")


def fast3d_text(status):
    """(text, is_problem) for a take's fast3d status: progress, and when done the calibration used and its check."""
    s = status.get("status")
    if s == "none":
        return "3D: not run", False
    if s == "queued":
        return "3D: queued", False
    if s == "processing":
        return f"3D: {status.get('stage', '')} {100 * status.get('progress', 0):.0f}%", False
    if s == "failed":
        return f"3D failed ({status.get('error', 'unknown error')})", True
    if s == "done_2d_only":
        return "3D: no matching calibration, 2D only -- pick a calibration", True
    cal = status.get("calibration") or {}
    name = os.path.basename(cal.get("path") or "?")
    verdict, med = cal.get("verdict"), cal.get("median_px")
    med_s = f"{med:.0f} px" if isinstance(med, (int, float)) else "?"
    if verdict == "ok":
        return f"3D done, calibration {name} OK ({med_s})", False
    flagged = ", ".join(cal.get("flagged") or [])
    what = "BAD" if verdict == "bad" else "WARNING"
    return f"3D done, calibration {name} {what} ({med_s}{'; check ' + flagged if flagged else ''})", True


class Fast3DWorker(threading.Thread):
    """Runs fast3d (python -m fast3d.run) for one take at a time, in a subprocess at lower priority, so the
    MediaPipe processes and their memory stay out of the app process and the GUI stays responsive. Separate from
    PostprocessWorker (grid video), so the two run side by side. Progress goes through the take's
    processed/fast3d_status.json, written by fast3d.run itself (read_fast3d_status above); the subprocess log is
    processed/fast3d/run.log.

    enqueue(take_dir, reuse_2d=False): reuse_2d reruns only the calibration choice and triangulation (about 25 s),
    e.g. after the take's calibration was changed.
    """

    def __init__(self, calib_dirs=()):
        super().__init__(daemon=True)
        self.queue = queue.Queue()
        self.calib_dirs = list(calib_dirs)

    def enqueue(self, take_dir, reuse_2d=False):
        take_dir = os.path.abspath(take_dir)
        os.makedirs(os.path.join(take_dir, "processed"), exist_ok=True)
        path = os.path.join(take_dir, FAST3D_STATUS)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"status": "queued"}, f)
        os.replace(path + ".tmp", path)
        self.queue.put((take_dir, reuse_2d))

    def run(self):
        import subprocess
        import sys
        while True:
            item = self.queue.get()
            if item is None:
                return
            take_dir, reuse_2d = item
            cmd = [sys.executable, "-m", "fast3d.run", take_dir]
            for d in self.calib_dirs:
                cmd += ["--calib-dir", os.path.abspath(d)]
            if reuse_2d and os.path.isfile(os.path.join(take_dir, "processed", "fast3d", "fast3d_2d.npz")):
                cmd.append("--reuse-2d")
            os.makedirs(os.path.join(take_dir, "processed", "fast3d"), exist_ok=True)
            kw = {}
            if os.name == "nt":
                kw["creationflags"] = 0x00004000          # BELOW_NORMAL_PRIORITY_CLASS
            else:
                kw["preexec_fn"] = lambda: os.nice(10)
            try:
                with open(os.path.join(take_dir, "processed", "fast3d", "run.log"), "w", encoding="utf-8") as log:
                    rc = subprocess.run(cmd, cwd=REPO_DIR, stdout=log, stderr=subprocess.STDOUT, **kw).returncode
                if rc != 0 and read_fast3d_status(take_dir).get("status") != "failed":
                    with open(os.path.join(take_dir, FAST3D_STATUS), "w", encoding="utf-8") as f:
                        json.dump({"status": "failed", "error": f"fast3d.run exited with {rc}, see processed/fast3d/run.log"}, f)
            except Exception as exc:
                print(f"[fast3d] {take_dir} failed: {exc}")
            self.queue.task_done()
