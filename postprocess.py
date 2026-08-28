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


class PostprocessWorker(threading.Thread):
    """One job at a time, serialized -- so postprocessing never competes with
    itself for CPU/ffmpeg, but a new take's recording is never blocked by a
    previous take's still-running postprocessing (see app.py's capture loop:
    jobs are only ever enqueued, never waited on before starting the next take).
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.queue = queue.Queue()

    def run(self):
        while True:
            take_dir = self.queue.get()
            if take_dir is None:
                return
            try:
                process_take(take_dir)
            except Exception as exc:
                print(f"[Postprocess] {take_dir} failed: {exc}")
            self.queue.task_done()
