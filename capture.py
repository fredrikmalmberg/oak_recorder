import argparse
import math
import os
import time
import queue
import threading
from datetime import datetime
import cv2
import depthai as dai
import numpy as np

from align_session import (
    align_session,
    convert_aligned_jpegs_to_grid_mp4,
    convert_mjpegs_to_grid_mp4,
    default_align_threshold_ms,
)

# ==============================================================================
# 1. OPTIMIZED CONFIGURATION FOR SIGN LANGUAGE KEYPOINT TRACKING
# ==============================================================================
FPS               = 30    # Target framerate
REC_W, REC_H      = 3840, 2160  # Pristine 4K for storage (MediaPipe / SMPLX)
VIEW_W, VIEW_H    = 1280, 720   # Lightweight 720p for GUI preview

# Hardware Encoder Settings
ENCODER_PROFILE   = dai.VideoEncoderProperties.Profile.MJPEG
MJPEG_QUALITY     = 70 # 95    # High JPEG quality to eliminate edge blurring

# Camera Exposure Controls (Completely eliminates motion blur)
# 1500 us = 1/666s shutter speed. Requires bright, flicker-free studio lights!
FORCED_SHUTTER_US = 2000  # max 2ms shutter speed
FORCED_ISO        = 200   # max 200 ISO
FORCED_WB_K       = 4500  # fixed white balance color temperature

RECORD_DIR        = "recordings"
PREVIEW_GRID_W    = 1920
PREVIEW_GRID_H    = 1080
PREVIEW_WINDOW    = "Sign Language Session Monitor"
DEFAULT_WARMUP_FRAMES = 60

# ==============================================================================
# 2. ASYNCHRONOUS FILE WRITER THREAD (Prevents GUI Frame Drops)
# ==============================================================================
class BackgroundVideoWriter(threading.Thread):
    """Append hardware-encoded JPEG frames to a raw MJPEG bytestream."""

    def __init__(self, filename, fps, width, height):
        super().__init__(daemon=True)
        self.frame_queue = queue.Queue(maxsize=240)
        self.running = True
        self.dropped = 0
        self.written = 0
        self.filename = filename
        self.fps = fps
        self.width = width
        self.height = height
        self._file = open(filename, "wb")

    def write_packet(self, packet_bytes):
        if isinstance(packet_bytes, (memoryview, bytearray)):
            packet_bytes = bytes(packet_bytes)
        elif hasattr(packet_bytes, "tobytes"):
            packet_bytes = packet_bytes.tobytes()
        try:
            self.frame_queue.put_nowait(packet_bytes)
        except queue.Full:
            self.dropped += 1

    def run(self):
        while self.running or not self.frame_queue.empty():
            try:
                data = self.frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            self._file.write(data)
            self.written += 1
            self.frame_queue.task_done()
        self._file.close()

    def stop(self):
        self.running = False
        self.join()


# ==============================================================================
# 3. PIPELINE GENERATOR (DepthAI v3 Architecture - Definitions Only)
# ==============================================================================
def create_dual_stream_pipeline(device, preview=True):
    pipeline = dai.Pipeline(device)

    cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam_rgb.initialControl.setManualExposure(FORCED_SHUTTER_US, FORCED_ISO)
    cam_rgb.initialControl.setManualWhiteBalance(FORCED_WB_K)

    out_rec = cam_rgb.requestOutput((REC_W, REC_H), type=dai.ImgFrame.Type.NV12, fps=FPS)

    video_enc = pipeline.create(dai.node.VideoEncoder).build(
        out_rec,
        frameRate=FPS,
        profile=ENCODER_PROFILE
    )
    video_enc.setQuality(MJPEG_QUALITY)

    out_preview = None
    if preview:
        out_preview = cam_rgb.requestOutput(
            (VIEW_W, VIEW_H), type=dai.ImgFrame.Type.BGR888p, fps=FPS
        )

    return pipeline, video_enc.out, out_preview


def session_dir_name(timestamp):
    return (f"sign_capture_{timestamp}_iso{FORCED_ISO}"
            f"_shutter{FORCED_SHUTTER_US}us")


def session_file_paths(session_dir, cam_idx):
    cam_label = f"cam{cam_idx}"
    video_path = os.path.join(session_dir, f"video_{cam_label}.mjpeg")
    log_path = os.path.join(session_dir, f"frame_timestamps_{cam_label}.log")
    return video_path, log_path


def compressed_mp4_path(video_path):
    session_dir = os.path.dirname(video_path)
    cam_label = os.path.basename(video_path).removeprefix("video_").removesuffix(".mjpeg")
    return os.path.join(session_dir, f"compressed_video_{cam_label}.mp4")


def xstack_layout(num_cameras):
    cols, _rows = preview_grid_layout(num_cameras)
    parts = []
    for i in range(num_cameras):
        row, col = divmod(i, cols)
        x = "0" if col == 0 else f"w{i - 1}"
        y = "0" if row == 0 else f"h{i - cols}"
        parts.append(f"{x}_{y}")
    return "|".join(parts)


def convert_mjpeg_to_mp4(video_path, fps):
    import subprocess

    mp4_path = compressed_mp4_path(video_path)
    cmd = [
        "ffmpeg",
        "-f", "mjpeg",
        "-framerate", str(fps),
        "-i", video_path,
        "-c:v", "libx264",
        "-r", str(fps),
        "-crf", "18",
        "-preset", "medium",
        "-tune", "film",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-y", mp4_path,
    ]
    subprocess.run(cmd, check=True)
    return mp4_path


def preview_grid_layout(num_cameras):
    cols = math.ceil(math.sqrt(num_cameras))
    rows = math.ceil(num_cameras / cols)
    return cols, rows


def compose_preview_grid(recorders):
    cols, rows = preview_grid_layout(len(recorders))
    cell_w = PREVIEW_GRID_W // cols
    cell_h = PREVIEW_GRID_H // rows
    grid = np.zeros((PREVIEW_GRID_H, PREVIEW_GRID_W, 3), dtype=np.uint8)

    for idx, recorder in enumerate(recorders):
        frame = recorder.preview_frame
        if frame is None:
            continue
        resized = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        row, col = divmod(idx, cols)
        y, x = row * cell_h, col * cell_w
        grid[y:y + cell_h, x:x + cell_w] = resized

    return grid


def show_preview(recorders):
    if len(recorders) == 1:
        frame = recorders[0].preview_frame
        if frame is not None:
            cv2.imshow(PREVIEW_WINDOW, frame)
        return

    cv2.imshow(PREVIEW_WINDOW, compose_preview_grid(recorders))


def drain_other_recorders(recorders, active_recorder):
    for recorder in recorders:
        if recorder is active_recorder:
            continue
        recorder.drain_rec_frame()
        recorder.drain_preview_frame()


def run_warmup(recorders, warmup_frames, preview_enabled):
    remaining = {r.cam_label: warmup_frames for r in recorders}
    print(f"[Warmup] Discarding {warmup_frames} frames per camera...")
    while any(remaining.values()):
        for recorder in recorders:
            if remaining[recorder.cam_label] <= 0:
                continue
            if recorder.drain_rec_frame():
                remaining[recorder.cam_label] -= 1
            if preview_enabled:
                recorder.drain_preview_frame()
        time.sleep(0.001)


class CameraRecorder:
    def __init__(self, cam_idx, device_info, session_dir, preview, record=True):
        self.cam_idx = cam_idx
        self.cam_label = f"cam{cam_idx}"
        self.record_enabled = record
        self.recording_active = False
        self.video_path = None
        self.log_path = None
        self.writer = None
        self.log_file = None
        self._drops_at_last_status = 0
        self.sync_host_ts_s = None
        self.sync_device_ts_s = None
        self.sync_host_offset_s = None

        if record:
            self.video_path, self.log_path = session_file_paths(session_dir, cam_idx)

        self.device = dai.Device(device_info)
        self.pipeline, rec_endpoint, view_endpoint = create_dual_stream_pipeline(
            self.device, preview=preview
        )
        self.q_rec = rec_endpoint.createOutputQueue(maxSize=8, blocking=False)
        self.q_view = None
        if view_endpoint is not None:
            self.q_view = view_endpoint.createOutputQueue(maxSize=4, blocking=False)
        self.pipeline.start()

        self.fps_counter = 0
        self.current_fps = 0.0
        self.preview_frame = None

    def begin_recording(self):
        if not self.record_enabled or self.recording_active:
            return
        self.writer = BackgroundVideoWriter(self.video_path, FPS, REC_W, REC_H)
        self.writer.start()
        self.log_file = open(self.log_path, "w", encoding="utf-8")
        self.log_file.write(f"# video={self.video_path}\n")
        self.log_file.write(f"# camera={self.cam_label}\n")
        self.log_file.write(f"# iso={FORCED_ISO} shutter_us={FORCED_SHUTTER_US} "
                            f"{REC_W}x{REC_H}@{FPS}fps mjpeg_q={MJPEG_QUALITY}\n")
        if self.sync_host_offset_s is not None:
            self.log_file.write(
                f"# sync_calibration host_ts_s={self.sync_host_ts_s:.9f} "
                f"device_ts_s={self.sync_device_ts_s:.9f} "
                f"host_offset_s={self.sync_host_offset_s:.9f}\n"
            )
            self.log_file.write("# sync_source=rec_mjpeg_post_warmup\n")
            self.log_file.write("# unified_time = device_timestamp_s + host_offset_s\n")
        self.log_file.write("# frame_log: host_ts_s sequence_num device_timestamp_s bytes\n")
        self.fps_counter = 0
        self._drops_at_last_status = 0
        self.recording_active = True

    def _rec_packet_bytes(self, rec_msg):
        raw_bytes = rec_msg.getData()
        if hasattr(raw_bytes, "tobytes"):
            return raw_bytes.tobytes()
        return bytes(raw_bytes)

    def flush_rec_and_preview(self):
        while self.drain_rec_frame():
            pass
        while self.drain_preview_frame():
            pass

    def capture_sync_calibration(self, session_dir):
        """Post-warmup MJPEG frame: map this camera device clock to host time."""
        self.flush_rec_and_preview()
        msg = self.q_rec.get(timeout=5.0)
        host_ts_s = time.time()
        ts = msg.getTimestamp()
        if ts is None:
            raise RuntimeError(
                f"{self.cam_label}: rec calibration frame missing device timestamp"
            )
        device_ts_s = ts.total_seconds()
        self.sync_host_ts_s = host_ts_s
        self.sync_device_ts_s = device_ts_s
        self.sync_host_offset_s = host_ts_s - device_ts_s

        still_path = os.path.join(session_dir, f"sync_still_{self.cam_label}.jpg")
        frame = cv2.imdecode(
            np.frombuffer(self._rec_packet_bytes(msg), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if frame is not None:
            cv2.imwrite(still_path, frame)
        print(
            f"[Sync] {self.cam_label}: rec-stream offset={self.sync_host_offset_s:.9f}s "
            f"host={host_ts_s:.9f} device={device_ts_s:.9f} "
            f"seq={msg.getSequenceNum()} -> {still_path}"
        )

    def drain_rec_frame(self):
        rec_msg = self.q_rec.tryGet()
        if rec_msg is None:
            return False
        return True

    def drain_preview_frame(self):
        if self.q_view is None:
            return False
        return self.q_view.tryGet() is not None

    def fetch_preview(self, preview_enabled):
        if not preview_enabled or self.q_view is None:
            return

        preview_msg = self.q_view.tryGet()
        if preview_msg is None:
            return

        frame = preview_msg.getCvFrame()
        rec_label = "REC: on" if self.recording_active else "REC: off"
        cv2.putText(frame, f"{self.cam_label} | {rec_label} @ {self.current_fps:.1f} FPS",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        self.preview_frame = frame

    def poll_record(self):
        rec_msg = self.q_rec.tryGet()
        if rec_msg is None:
            return

        host_ts_s = time.time()
        if self.recording_active:
            data = self._rec_packet_bytes(rec_msg)
            self.writer.write_packet(data)
            ts = rec_msg.getTimestamp()
            device_ts_s = ts.total_seconds() if ts is not None else float("nan")
            self.log_file.write(
                f"{host_ts_s:.9f} {rec_msg.getSequenceNum()} "
                f"{device_ts_s:.9f} {len(data)}\n"
            )
            self.fps_counter += 1
        elif not self.record_enabled:
            self.fps_counter += 1

    def update_fps(self, elapsed):
        self.current_fps = self.fps_counter / elapsed if elapsed > 0 else 0.0
        self.fps_counter = 0

    def status_line(self):
        return f"{self.cam_label}: {self.current_fps:.1f} FPS"

    def new_drops_since_last_status(self):
        if not self.recording_active or self.writer is None:
            return 0
        return self.writer.dropped - self._drops_at_last_status

    def mark_status_interval(self):
        if self.writer is not None:
            self._drops_at_last_status = self.writer.dropped

    def close(self):
        if self.log_file is not None:
            self.log_file.close()
        if self.writer is not None:
            self.writer.stop()
        try:
            self.pipeline.stop()
        except Exception:
            pass
        self.device.close()


# ==============================================================================
# 4. MAIN CAPTURE AND RENDERING LOOP
# ==============================================================================
def main(args=None):
    os.makedirs(RECORD_DIR, exist_ok=True)
    if args is not None:
        try:
            global FORCED_ISO, FORCED_SHUTTER_US, FPS
            FORCED_ISO = int(args.iso)
            FORCED_SHUTTER_US = int(args.shutter)
            FPS = int(args.fps)
        except Exception:
            pass

    no_preview = args.no_preview if args is not None else False
    preview_enabled = not no_preview
    record_enabled = not (args.no_record if args is not None else False)
    warmup_enabled = not (args.no_warmup if args is not None else False)
    align_enabled = not (args.no_align if args is not None else False)
    warmup_frames = args.warmup_frames if args is not None else DEFAULT_WARMUP_FRAMES
    align_threshold_ms = (
        args.align_threshold_ms if args is not None and args.align_threshold_ms is not None
        else default_align_threshold_ms(FPS)
    )
    align_host_only = args.align_host if args is not None else False
    align_raw = args.align_raw if args is not None else False
    output_mp4 = args.output_mp4 if args is not None else None
    if align_raw:
        align_enabled = False
        if output_mp4 is None:
            output_mp4 = "small"
    if not record_enabled:
        output_mp4 = None
        align_enabled = False
        align_raw = False

    print("Initializing OAK device(s)...")

    device_infos = dai.Device.getAllAvailableDevices()
    if not device_infos:
        print("[Error] No active OAK device discovered.")
        return

    print(f"Found {len(device_infos)} OAK device(s):")
    for cam_idx, info in enumerate(device_infos):
        print(f"  [{cam_idx}] ID: {info.deviceId}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = None
    if record_enabled:
        session_dir = os.path.join(RECORD_DIR, session_dir_name(timestamp))
        os.makedirs(session_dir, exist_ok=True)
    recorders = []

    try:
        for cam_idx, device_info in enumerate(device_infos):
            print(f"Starting cam{cam_idx}...")
            recorders.append(CameraRecorder(
                cam_idx, device_info, session_dir, preview_enabled, record_enabled
            ))
    except Exception as e:
        print(f"[Error] Failed to start camera: {e}")
        for recorder in recorders:
            recorder.close()
        return

    print(f"\n[Recording Setup]")
    if record_enabled:
        print(f"  Session folder: {session_dir}")
    else:
        print("  Recording: disabled")
    print(f"  Configuration: 4K ({REC_W}x{REC_H}) MJPEG @ {FPS}fps")
    print(f"  Shutter Time: {FORCED_SHUTTER_US} us | ISO: {FORCED_ISO}")
    print(f"  Warmup: {'enabled' if warmup_enabled and record_enabled else 'disabled'}"
          + (f" ({warmup_frames} frames)" if warmup_enabled and record_enabled else ""))
    if align_raw:
        print("  Alignment: raw MP4 only (no timestamp alignment)")
    elif align_enabled:
        align_mode = "host clock" if align_host_only else "unified device"
        print(f"  Alignment: enabled ({align_mode}, threshold {align_threshold_ms:.1f}ms)")
    else:
        print("  Alignment: disabled")
    if output_mp4 == "small":
        if align_raw:
            src = "raw MJPEG"
        else:
            src = "aligned JPEGs" if align_enabled else "raw MJPEG"
        cols, rows = preview_grid_layout(len(device_infos))
        print(f"  MP4 export: small grid ({cols}x{rows} @ {PREVIEW_GRID_W}x{PREVIEW_GRID_H}) "
              f"from {src}")
    elif output_mp4 == "actual":
        print(f"  MP4 export: enabled ({REC_W}x{REC_H}, full resolution per camera)")
    else:
        print("  MP4 export: disabled")
    if record_enabled:
        for recorder in recorders:
            print(f"  {recorder.cam_label}: {recorder.video_path}")
    if no_preview:
        print("  Preview: disabled")
        print("  Press Ctrl+C in the terminal to stop capture.\n")
    else:
        if len(recorders) > 1:
            cols, rows = preview_grid_layout(len(recorders))
            print(f"  Preview: {cols}x{rows} grid @ {PREVIEW_GRID_W}x{PREVIEW_GRID_H}")
        print("  Press 'q' in the preview window to stop capture cleanly.\n")

    if record_enabled:
        if warmup_enabled:
            run_warmup(recorders, warmup_frames, preview_enabled)
        print("[Sync] Calibrating device-host offset from rec stream (post-warmup)...")
        for recorder in recorders:
            drain_other_recorders(recorders, recorder)
            recorder.capture_sync_calibration(session_dir)
        for recorder in recorders:
            recorder.begin_recording()
        print(f"[Recording started] {datetime.now().isoformat()}")

    fps_timestamp = time.monotonic()

    try:
        while True:
            for recorder in recorders:
                recorder.fetch_preview(preview_enabled)
                recorder.poll_record()

            if preview_enabled:
                show_preview(recorders)

            now = time.monotonic()
            if now - fps_timestamp >= 1.0:
                elapsed = now - fps_timestamp
                for recorder in recorders:
                    recorder.update_fps(elapsed)
                status_parts = [r.status_line() for r in recorders]
                drop_parts = []
                for r in recorders:
                    drops = r.new_drops_since_last_status()
                    if drops > 0:
                        drop_parts.append(f"{r.cam_label} DROPPED: {drops}")
                    r.mark_status_interval()
                line = "[Status] " + " | ".join(status_parts)
                if drop_parts:
                    line += " | " + " | ".join(drop_parts)
                print(line)
                fps_timestamp = now

            if no_preview:
                time.sleep(0.001)
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nCapture manually interrupted by user via terminal.")

    finally:
        print("\nShutting down pipeline(s) and finalizing files...")
        if preview_enabled:
            cv2.destroyAllWindows()

        video_paths = []
        cam_labels = []
        for recorder in recorders:
            recorder.close()
            if recorder.writer and recorder.writer.dropped > 0:
                print(f"[Warning] {recorder.cam_label}: {recorder.writer.dropped} "
                      f"frames dropped (writer queue full).")
            if record_enabled:
                print(f"  {recorder.cam_label} video: {recorder.video_path}")
                print(f"  {recorder.cam_label} log: {recorder.log_path}")
                video_paths.append(recorder.video_path)
                cam_labels.append(recorder.cam_label)

        alignment_ran = False
        if align_enabled and video_paths:
            try:
                align_session(
                    session_dir,
                    align_threshold_ms=align_threshold_ms,
                    fps=float(FPS),
                    rec_w=REC_W,
                    rec_h=REC_H,
                    align_host_only=align_host_only,
                )
                alignment_ran = True
            except Exception as e:
                print(f"[Error] Alignment failed: {e}")

        if output_mp4 == "actual" and video_paths:
            for recorder in recorders:
                if not recorder.video_path:
                    continue
                try:
                    mp4_path = convert_mjpeg_to_mp4(recorder.video_path, FPS)
                    print(f"[Success] {recorder.cam_label} saved as mp4: {mp4_path}")
                except Exception as e:
                    print(f"[Error] {recorder.cam_label} failed to save mp4: {e}")
        elif output_mp4 == "small" and video_paths:
            try:
                if alignment_ran:
                    mp4_path = convert_aligned_jpegs_to_grid_mp4(
                        session_dir, cam_labels, FPS
                    )
                else:
                    mp4_path = convert_mjpegs_to_grid_mp4(video_paths, FPS, session_dir)
                print(f"[Success] Grid mp4 saved: {mp4_path}")
            except Exception as e:
                print(f"[Error] Failed to save grid mp4: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign Language Capture")
    parser.add_argument("-i", "--iso", type=int, default=200, help="ISO value")
    parser.add_argument("-s", "--shutter", type=int, default=10000,
                        help="Shutter speed in microseconds")
    parser.add_argument("-f", "--fps", type=int, default=30, help="Frames per second")
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable preview window and skip the 720p camera stream",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable saving MJPEG files and MP4 conversion",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip pre-capture warmup frame discard",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=DEFAULT_WARMUP_FRAMES,
        help="Frames to discard per camera during warmup",
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Skip post-capture timestamp alignment",
    )
    parser.add_argument(
        "--align-threshold-ms",
        type=float,
        default=None,
        help="Max timestamp offset (ms) to accept frame match during alignment",
    )
    parser.add_argument(
        "--align-host",
        action="store_true",
        help="Align on raw host receive timestamps (ignore device sync calibration)",
    )
    parser.add_argument(
        "--align-raw",
        action="store_true",
        help="Skip alignment; build grid MP4 from raw MJPEG only",
    )
    parser.add_argument(
        "--output-mp4",
        choices=["small", "actual"],
        default=None,
        help="Convert MJPEG to MP4 after capture. "
             "'small' writes one grid MP4, 'actual' writes full-resolution MP4 per camera.",
    )
    args = parser.parse_args()
    if args.align_raw and args.align_host:
        parser.error("--align-raw cannot be combined with --align-host")

    main(args)
