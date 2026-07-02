import argparse
import os
import time
import queue
import threading
from datetime import datetime
import cv2
import depthai as dai

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

RECORD_DIR        = "recordings"

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

    # Create the unified v3 Camera Node
    cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    
    # Apply raw manual exposure constraints to eliminate auto-exposure shifting
    cam_rgb.initialControl.setManualExposure(FORCED_SHUTTER_US, FORCED_ISO)

    # STREAM A: 4K NV12 -> On-Device MJPEG Encoder Endpoint
    out_rec = cam_rgb.requestOutput((REC_W, REC_H), type=dai.ImgFrame.Type.NV12, fps=FPS)
    
    video_enc = pipeline.create(dai.node.VideoEncoder).build(
        out_rec, 
        frameRate=FPS, 
        profile=ENCODER_PROFILE
    )
    video_enc.setQuality(MJPEG_QUALITY)

    # STREAM B: 720p Raw BGR Endpoint (skipped when preview is disabled)
    out_preview = None
    if preview:
        out_preview = cam_rgb.requestOutput(
            (VIEW_W, VIEW_H), type=dai.ImgFrame.Type.BGR888p, fps=FPS
        )
    
    return pipeline, video_enc.out, out_preview


# ==============================================================================
# 4. MAIN CAPTURE AND RENDERING LOOP
# ==============================================================================
def main(args=None):
    os.makedirs(RECORD_DIR, exist_ok=True)
    # Apply CLI overrides to the module-level configuration when provided
    if args is not None:
        try:
            # update globals so downstream functions use the requested values
            global FORCED_ISO, FORCED_SHUTTER_US, FPS
            FORCED_ISO = int(args.iso)
            FORCED_SHUTTER_US = int(args.shutter)
            FPS = int(args.fps)
        except Exception:
            # If conversion fails, continue with defaults
            pass
    
    no_preview = args.no_preview if args is not None else False

    print("Initializing OAK Device...")
    
    # Unpack the (status_bool, device_info) tuple returned by the API
    success, device_info = dai.Device.getFirstAvailableDevice()
    if not success or device_info is None:
        print("[Error] No active OAK device discovered.")
        return

    # Pass the isolated DeviceInfo object to the constructor
    device = dai.Device(device_info)
    pipeline, rec_endpoint, view_endpoint = create_dual_stream_pipeline(
        device, preview=not no_preview
    )
    
    # FIX: Initialize and hook the Output Queues BEFORE running pipeline.start()
    q_rec = rec_endpoint.createOutputQueue(maxSize=8, blocking=False)
    q_view = None
    if view_endpoint is not None:
        q_view = view_endpoint.createOutputQueue(maxSize=4, blocking=False)
    
    # Now start the hardware pipeline execution state safely
    pipeline.start()

    # Generate unique structured session file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = (f"sign_capture_{timestamp}_iso{FORCED_ISO}"
                  f"_shutter{FORCED_SHUTTER_US}us.mjpeg")
    video_path = os.path.join(RECORD_DIR, video_name)
    log_path   = os.path.join(RECORD_DIR,
                              f"sign_capture_{timestamp}_iso{FORCED_ISO}"
                              f"_shutter{FORCED_SHUTTER_US}us.log")

    print(f"\n[Recording Setup]")
    print(f"  Destination: {video_path}")
    print(f"  Configuration: 4K ({REC_W}x{REC_H}) MJPEG @ {FPS}fps")
    print(f"  Shutter Time: {FORCED_SHUTTER_US} us | ISO: {FORCED_ISO}")
    if no_preview:
        print("  Preview: disabled (4K record stream only)")
        print("  Press Ctrl+C in the terminal to stop capture.\n")
    else:
        print("  Press 'q' in the preview window to stop capture cleanly.\n")

    # Spin up the threaded file handling writer context
    writer = BackgroundVideoWriter(video_path, FPS, REC_W, REC_H)
    writer.start()

    # Open structural CSV/Log file for frame profiling validation
    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write(f"# video={video_path}\n")
    log_file.write(f"# iso={FORCED_ISO} shutter_us={FORCED_SHUTTER_US} "
                   f"{REC_W}x{REC_H}@{FPS}fps mjpeg_q={MJPEG_QUALITY}\n")
    log_file.write("# host_timestamp sequence_num device_timestamp_s bytes\n")

    frame_count = 0
    fps_timestamp = time.monotonic()
    fps_counter = 0
    current_fps = 0.0

    try:
        while True:
            if q_view is not None:
                preview_msg = q_view.tryGet()
                if preview_msg is not None:
                    frame = preview_msg.getCvFrame()
                    cv2.putText(frame, f"REC: 4K | VIEW: 720p @ {current_fps:.1f} FPS", (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, f"Frames Logged: {frame_count} | Dropped: {writer.dropped}", (30, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow("Sign Language Session Monitor", frame)

            # Pull the pristine 4K compressed MJPEG frame packet out for storage
            rec_msg = q_rec.tryGet()
            if rec_msg is not None:
                raw_bytes = rec_msg.getData()
                
                data = (raw_bytes.tobytes() if hasattr(raw_bytes, "tobytes")
                        else bytes(raw_bytes))
                writer.write_packet(data)

                frame_count += 1
                fps_counter += 1
                ts = rec_msg.getTimestamp()
                ts_s = f"{ts.total_seconds():.9f}" if ts is not None else "n/a"
                log_file.write(f"{datetime.now().isoformat()} "
                               f"{rec_msg.getSequenceNum()} {ts_s} {len(data)}\n")

            now = time.monotonic()
            if now - fps_timestamp >= 1.0:
                current_fps = fps_counter / (now - fps_timestamp)
                if no_preview:
                    print(f"[Status] {current_fps:.1f} FPS | "
                          f"Frames: {frame_count} | Dropped: {writer.dropped}")
                fps_counter = 0
                fps_timestamp = now

            if no_preview:
                time.sleep(0.001)
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nCapture manually interrupted by user via terminal.")

    finally:
        print("\nShutting down pipeline and finalizing files...")
        log_file.close()
        writer.stop()
        try:
            pipeline.stop()
        except Exception:
            pass
        if not no_preview:
            cv2.destroyAllWindows()
        device.close()
        
        print(f"[Done] Frames captured: {frame_count}  written: {writer.written}")
        if writer.dropped > 0:
            print(f"[Warning] {writer.dropped} frames dropped (writer queue full).")
        print(f"  Video Asset: {video_path}")
        print(f"  Session Profile Log: {log_path}")

        # try to save the file as mp4 using ffmpeg
        try:
            import subprocess
            mp4_path = video_path.replace(".mjpeg", ".mp4")
            subprocess.run([
                "ffmpeg",
                "-f", "mjpeg",
                "-framerate", str(FPS),
                "-i", video_path,
                "-c:v", "libx264",
                "-r", str(FPS),
                "-crf", "18",
                "-preset", "medium",
                "-tune", "film",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-y", mp4_path,
            ])
            print(f"[Success] File saved as mp4: {mp4_path}")
        except Exception as e:
            print(f"[Error] Failed to save file as mp4: {e}")

if __name__ == "__main__":
    # add command line argument parsing
    parser = argparse.ArgumentParser(description="Sign Language Capture")
    parser.add_argument("-i", "--iso", type=int, default=200, help="ISO value")
    parser.add_argument("-s", "--shutter", type=int, default=10000, help="Shutter speed in microseconds")
    parser.add_argument("-f", "--fps", type=int, default=30, help="Frames per second")
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable preview window and skip the 720p camera stream",
    )
    args = parser.parse_args()

    main(args)
