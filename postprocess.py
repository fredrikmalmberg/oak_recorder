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

from align_session import convert_mjpegs_to_grid_mp4


def _step_grid_video(take_dir, meta):
    processed_dir = os.path.join(take_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    video_paths = [cam["video_path"] for cam in meta["cameras"].values()]
    convert_mjpegs_to_grid_mp4(video_paths, meta["fps"], processed_dir)


PROCESSING_STEPS = [_step_grid_video]


def process_take(take_dir):
    with open(os.path.join(take_dir, "take_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    for step in PROCESSING_STEPS:
        step(take_dir, meta)


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


def _write_status(take_dir, status, error=None):
    payload = {"status": status}
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
                process_take(take_dir)
            except Exception as exc:
                print(f"[Postprocess] {take_dir} failed: {exc}")
                _write_status(take_dir, "failed", error=exc)
            else:
                _write_status(take_dir, "done")
            self.queue.task_done()
