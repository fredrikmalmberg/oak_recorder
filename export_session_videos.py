"""One-off export: converts every camera's raw MJPEG to an individual MP4
for the given sessions, and copies each take's already-rendered grid video
alongside them -- into exports/<session_ts>/<take_name>/, flat and easy to
grab over FTP. Not part of the app's own pipeline (capture.py's
convert_mjpeg_to_mp4 already does per-camera conversion, but always writes
next to the source MJPEG; this just points the same ffmpeg settings at a
separate export tree instead).
"""
import json
import os
import shutil
import subprocess
import sys
import time

SESSIONS_DIR = "recordings"
EXPORT_DIR = "exports"


def convert_camera_mp4(mjpeg_path, fps, out_path):
    cmd = [
        "ffmpeg",
        "-f", "mjpeg",
        "-framerate", str(fps),
        "-i", mjpeg_path,
        "-c:v", "libx264",
        "-r", str(fps),
        "-crf", "18",
        "-preset", "medium",
        "-tune", "film",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-y", out_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def export_session(session_ts):
    session_dir = os.path.join(SESSIONS_DIR, session_ts)
    take_names = sorted(
        d for d in os.listdir(session_dir)
        if os.path.isdir(os.path.join(session_dir, d)) and d.startswith("take_")
    )
    for take_name in take_names:
        take_dir = os.path.join(session_dir, take_name)
        meta_path = os.path.join(take_dir, "take_meta.json")
        if not os.path.exists(meta_path):
            print(f"  [skip] {take_dir}: no take_meta.json")
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        fps = meta["fps"]

        out_dir = os.path.join(EXPORT_DIR, session_ts, take_name)
        os.makedirs(out_dir, exist_ok=True)

        for cam_id, cam_meta in meta["cameras"].items():
            mjpeg_path = cam_meta["video_path"]
            out_path = os.path.join(out_dir, f"{cam_id}.mp4")
            if os.path.exists(out_path):
                print(f"  [skip] {out_path} already exists")
                continue
            t0 = time.monotonic()
            print(f"  [convert] {mjpeg_path} -> {out_path} (fps={fps}) ...", flush=True)
            convert_camera_mp4(mjpeg_path, fps, out_path)
            print(f"    done in {time.monotonic() - t0:.0f}s", flush=True)

        grid_src = os.path.join(take_dir, "processed", "compressed_video_grid.mp4")
        grid_dst = os.path.join(out_dir, "grid.mp4")
        if os.path.exists(grid_src):
            if not os.path.exists(grid_dst):
                shutil.copy2(grid_src, grid_dst)
                print(f"  [copy] {grid_src} -> {grid_dst}")
        else:
            print(f"  [warn] no grid video found at {grid_src}")


if __name__ == "__main__":
    sessions = sys.argv[1:]
    if not sessions:
        print("usage: export_session_videos.py <session_ts> [<session_ts> ...]")
        sys.exit(1)
    for session_ts in sessions:
        print(f"=== session {session_ts} ===", flush=True)
        export_session(session_ts)
    print("ALL DONE")
