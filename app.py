"""Combined calibration + capture UI (viser). Boots every discovered OAK
camera exactly once (reusing capture.py's hardened connect_device_with_retry
-- stagger, retries, boot-log JSONL -- for the same reason capture.py needed
it: OAK cameras draw a current spike while booting, and several connecting
close together can brown out whichever one boots last), then lets the
operator load a saved calibration, run a live one, or continue uncalibrated
-- all without re-booting cameras between those choices.

Boot order is NOT random: camera_boot_stats.rank_boot_order sorts known
troublesome cameras (by past failure count, from logs/camera_boot_log.jsonl)
to the FRONT of the queue, since historical data showed failures cluster
almost entirely at the LAST boot position -- a camera with a bad track
record should never be the one left competing against every other
already-connected camera for power.

Calibration mode reuses calibrate.py's engine directly (build_board,
CameraCalibState, process_detection, PoseGraph, world alignment, intrinsics
cache, and the whole ViserManager) rather than reimplementing any of it --
see the module docstring in calibrate.py for the underlying design notes.

Capture mode does NOT swap to a different pipeline: there is exactly one
dai.Pipeline per camera for the whole app lifetime (see
AppCameraSession.start_calibration_pipeline), used for both live ChArUco
detection AND recording, at whatever cfg["camera"]["fps"]/["mjpeg_quality"]
say. Two reasons: building a second dai.Pipeline on an already-used,
already-stopped dai.Device was found to crash depthai 3.7.1 natively
(confirmed via isolated hardware testing), and swapping pipelines the
"safe" way (close+reopen every device) reintroduces the exact same
multi-camera boot/brownout risk this module's own boot_cameras step exists
to harden against -- reproduced live (the same historically-bad camera
failed to reconnect) before this design was simplified to avoid it
entirely. So entering capture mode is instant: a take just starts writing
the already-running stream to disk. See README.md's "Capture mode" section
for the full session/take/postprocessing design.
"""
import argparse
import os
import queue
import re
import threading
import time
from datetime import datetime

import cv2
import numpy as np

try:
    import depthai as dai
except ImportError:
    dai = None

import json

try:
    import psutil
except ImportError:
    psutil = None

import calibrate
import camera_boot_stats
import capture
import postprocess

CALIBRATION_FILENAME_RE = re.compile(r"^\d{8}_\d{6}_\d+cam\.json$")


def list_available_calibrations(cfg):
    """Saved calibrate.py session outputs under cfg["output"]["dir"], newest
    first -- filtered to calibration_output_path's own naming convention so
    intrinsics_cache.json (same directory) and any stray files are excluded.
    """
    out_dir = cfg["output"]["dir"]
    if not os.path.isdir(out_dir):
        return []
    paths = [
        os.path.join(out_dir, name) for name in os.listdir(out_dir)
        if CALIBRATION_FILENAME_RE.match(name)
    ]
    return sorted(paths, reverse=True)


def matches_active_cameras(path, active_device_ids):
    """Exact match, not "the file's cameras are a superset of what's active"
    -- a saved calibration missing even one currently-connected camera, or
    covering extra cameras that aren't connected right now, isn't a real
    match for this session. Reuses load_calibration_output's own filtering
    (only cameras that actually got a solved pose count), so this always
    agrees with what loading the file would actually produce.
    """
    loaded = calibrate.load_calibration_output(path)
    saved_device_ids = {entry["device_id"] for entry in loaded.values() if entry.get("device_id")}
    return saved_device_ids == active_device_ids, len(saved_device_ids)


def verify_floor_board(viser_mgr, sessions, cam_ids, loaded_cameras_candidate,
                        alignment_detector, alignment_board_points_3d, cfg):
    """Pre-flight sanity check before "Load & start capture" commits: tries
    to detect the alignment/floor board live, using each camera's OWN saved
    intrinsics from the candidate file, and shows the PnP reprojection
    error -- a large jump from what's expected suggests the rig has moved
    since this calibration was saved. Runs synchronously on the setup
    thread (the same one already blocking on the picker) -- nothing else is
    happening at this point in the app's life, so this is the "background"
    check without needing real threading. Returns True to proceed with
    loading, False to go back to the picker (operator hit Cancel with no
    board detected).
    """
    with viser_mgr.server.gui.add_modal("Verifying calibration...") as modal:
        status_md = viser_mgr.server.gui.add_markdown(
            "_Looking for the floor board -- place it where any camera can see it..._"
        )
        ok_btn = viser_mgr.server.gui.add_button("OK", visible=False)
        proceed_btn = viser_mgr.server.gui.add_button("Proceed anyway", visible=False)
        cancel_btn = viser_mgr.server.gui.add_button("Cancel", visible=False)

    proceed_result = {"value": None}
    done = threading.Event()

    def _ok(_):
        proceed_result["value"] = True
        done.set()

    def _proceed_anyway(_):
        proceed_result["value"] = True
        done.set()

    def _cancel(_):
        proceed_result["value"] = False
        done.set()

    ok_btn.on_click(_ok)
    proceed_btn.on_click(_proceed_anyway)
    cancel_btn.on_click(_cancel)

    found_cam_id, found_err = None, None
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and found_cam_id is None:
        for cam_id in cam_ids:
            session = sessions[cam_id]
            if not session.connected:
                continue
            entry = loaded_cameras_candidate.get(session.device_id)
            if entry is None:
                continue
            try:
                got = session.poll(decode_full=True, decode_preview=False)
            except Exception:
                continue
            if not got or session.last_frame_full is None:
                continue
            gray = cv2.cvtColor(session.last_frame_full, cv2.COLOR_BGR2GRAY)
            detection = calibrate.detect_charuco(alignment_detector, gray)
            if detection is None:
                continue
            corners2d, ids = detection
            if len(ids) < cfg["quality_gates"]["min_corners"]:
                continue
            solved = calibrate.solve_board_pose(
                alignment_board_points_3d[ids], corners2d, entry["K"], entry["dist"],
            )
            if solved is None:
                continue
            _R, _t, found_err = solved
            found_cam_id = cam_id
            break
        if found_cam_id is None:
            time.sleep(0.02)

    if found_cam_id is not None:
        status_md.content = (
            f"_Floor board detected by **{found_cam_id}**._\n\n"
            f"**Reprojection error: {found_err:.3f}px**\n\n"
            "Does this look right (rig hasn't moved much since this calibration "
            "was saved)?"
        )
        ok_btn.visible = True
    else:
        status_md.content = (
            "_No floor board detected within a couple seconds._\n\n"
            "Proceed anyway without this check?"
        )
        proceed_btn.visible = True
        cancel_btn.visible = True

    done.wait()
    modal.close()
    return proceed_result["value"]


def maybe_undistort(frame, K, dist, display_size, record_size):
    """K=None means "no usable intrinsics for this camera right now" (not
    yet converged / not in the loaded file / uncalibrated mode) -- returns
    frame unchanged rather than raising, which is what lets the global
    undistort toggle apply blindly across every camera regardless of
    whether each one actually has intrinsics yet.

    K is computed for the FULL recording resolution, but frame is at the
    smaller preview resolution -- fx/fy/cx/cy scale with resolution, so K
    needs rescaling before cv2.undistort can be applied here; dist
    coefficients don't (OpenCV's pinhole model is scale-invariant there).
    """
    if K is None:
        return frame
    scale = display_size[0] / record_size[0]
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale
    K_scaled[1, 1] *= scale
    K_scaled[0, 2] *= scale
    K_scaled[1, 2] *= scale
    return cv2.undistort(frame, K_scaled, dist)


# ==============================================================================
# Camera session: connects once (via capture.py's hardened retry helper) and
# holds whichever pipeline is currently active on that connection. Deliberately
# NOT calibrate.CalibCameraSession, which bundles a plain dai.Device() connect
# with pipeline start in one call -- that bundling is exactly what this app
# needs to avoid, since the whole point is booting once and swapping pipelines.
# ==============================================================================
class AppCameraSession:
    def __init__(self, cam_id, device_info, session_ts, boot_position, cfg):
        self.cam_id = cam_id
        self.cam_label = cam_id  # capture.close_recorders_parallel expects this attribute
        self.device_info = device_info
        self.device_id = device_info.deviceId
        self.session_ts = session_ts
        self.boot_position = boot_position
        self.cfg = cfg
        self.usb_speed = None
        self.device = None
        self.pipeline = None
        self.q_full = None
        self.q_preview = None
        self.control_queue = None
        self.last_frame_full = None
        self.last_frame_full_ts = None
        self.last_frame_preview = None
        self.connected = False
        self.fps = None  # exponentially-smoothed full-frame arrival rate, see poll()
        self._fps_prev_mono = None
        self.actual_shutter_us = None  # last-reported actually-applied exposure, see poll()
        self.actual_iso = None
        self.recording_active = False
        self.video_path = None
        self.log_path = None
        self.writer = None  # capture.BackgroundVideoWriter, while recording_active
        self.log_file = None
        self.sync_host_ts_s = None
        self.sync_device_ts_s = None
        self.sync_host_offset_s = None
        self.preview_updated = False  # set each poll() call, see poll()'s decode_preview

    def connect(self):
        self.device, self.usb_speed = capture.connect_device_with_retry(
            self.device_info, self.session_ts, self.boot_position,
        )
        self.connected = True

    def start_calibration_pipeline(self):
        """Mirrors the tail of calibrate.CalibCameraSession.__init__ -- same
        pipeline shape (build_calibration_pipeline), just decoupled from the
        connect step above. with_control=True additionally gets a control
        queue for live ISO/shutter changes (see set_exposure) -- calibrate.py
        itself doesn't need this, so it stays opt-in there.

        This is the ONLY pipeline this session ever builds -- capture mode
        does not swap to a different one (see begin_recording): building a
        second dai.Pipeline on an already-used, already-stopped dai.Device
        was found to crash depthai 3.7.1 natively (confirmed via isolated
        hardware testing), and the operator explicitly wants no reconnect
        cascade when entering capture mode, so capture just starts writing
        this same running pipeline's full-res stream to disk instead.
        q_full's maxSize=8 (not calibrate.py's own maxSize=4) accounts for
        it also being the recording source once a take is active -- dropped
        frames there would mean silently missing frames in the recording,
        not just a stale preview.
        """
        self.pipeline, full_endpoint, preview_endpoint, self.control_queue = (
            calibrate.build_calibration_pipeline(self.device, self.cfg, with_control=True)
        )
        self.q_full = full_endpoint.createOutputQueue(maxSize=8, blocking=False)
        self.q_preview = preview_endpoint.createOutputQueue(maxSize=2, blocking=False)
        self.pipeline.start()

    def set_exposure(self, shutter_us, iso):
        """Live exposure change -- initialControl (set at pipeline build time)
        only affects the starting value, so this is the only way to change
        it after pipeline.start() without rebuilding the pipeline.
        """
        if self.control_queue is None:
            return
        ctrl = dai.CameraControl()
        ctrl.setManualExposure(shutter_us, iso)
        self.control_queue.send(ctrl)

    def begin_recording(self, video_path, log_path, header_extra_lines=()):
        """Ports CameraRecorder.begin_recording (capture.py) -- same header
        shape (video path; camera/device_id/usb_speed; iso/shutter/
        resolution/fps/quality; sync-calibration block if available; column
        header), plus caller-supplied extra lines (take/session/calibration
        info) so this method itself doesn't need to know about that.

        fps/mjpeg_quality come from cfg["camera"] -- there's only one
        pipeline for the whole app lifetime (see start_calibration_pipeline),
        so recording writes whatever it's already producing, not a separate
        capture-tuned config.
        """
        cam_cfg = self.cfg["camera"]
        self.video_path, self.log_path = video_path, log_path
        self.writer = capture.BackgroundVideoWriter(
            video_path, cam_cfg["fps"], cam_cfg["record_width"], cam_cfg["record_height"],
        )
        self.writer.start()
        self.log_file = open(log_path, "w", encoding="utf-8")
        self.log_file.write(f"# video={video_path}\n")
        self.log_file.write(f"# camera={self.cam_label} device_id={self.device_id} "
                             f"usb_speed={self.usb_speed.name}\n")
        self.log_file.write(f"# iso={self.actual_iso} shutter_us={self.actual_shutter_us} "
                             f"{cam_cfg['record_width']}x{cam_cfg['record_height']}@{cam_cfg['fps']}fps "
                             f"mjpeg_q={cam_cfg['mjpeg_quality']}\n")
        if self.sync_host_offset_s is not None:
            self.log_file.write(
                f"# sync_calibration host_ts_s={self.sync_host_ts_s:.9f} "
                f"device_ts_s={self.sync_device_ts_s:.9f} "
                f"host_offset_s={self.sync_host_offset_s:.9f}\n"
            )
            self.log_file.write("# sync_source=rec_mjpeg_per_take\n")
            self.log_file.write("# unified_time = device_timestamp_s + host_offset_s\n")
        for line in header_extra_lines:
            self.log_file.write(f"# {line}\n")
        self.log_file.write("# frame_log: host_ts_s sequence_num device_timestamp_s bytes\n")
        self.recording_active = True

    def capture_sync_calibration(self, take_dir):
        """Ports CameraRecorder.capture_sync_calibration (capture.py) -- one
        blocking full-res frame read maps this camera's device clock to host
        time. Run once per TAKE (not once per session): cheap (a single
        frame), and sidesteps the untested long-idle-drift question the
        README's persistent-streams note flags for an open pipeline sitting
        idle between takes -- each take gets a fresh offset instead.
        """
        while self.q_full.tryGet() is not None:
            pass
        while self.q_preview.tryGet() is not None:
            pass
        msg = self.q_full.get(timeout=5.0)
        host_ts_s = time.time()
        ts = msg.getTimestamp()
        if ts is None:
            raise RuntimeError(f"{self.cam_label}: sync-calibration frame missing device timestamp")
        device_ts_s = ts.total_seconds()
        self.sync_host_ts_s = host_ts_s
        self.sync_device_ts_s = device_ts_s
        self.sync_host_offset_s = host_ts_s - device_ts_s

        still_path = os.path.join(take_dir, f"sync_still_{self.cam_label}.jpg")
        frame = cv2.imdecode(
            np.frombuffer(calibrate._packet_bytes(msg), dtype=np.uint8), cv2.IMREAD_COLOR,
        )
        if frame is not None:
            cv2.imwrite(still_path, frame)
        print(f"[Sync] {self.cam_label}: offset={self.sync_host_offset_s:.9f}s "
              f"host={host_ts_s:.9f} device={device_ts_s:.9f} "
              f"seq={msg.getSequenceNum()} -> {still_path}")

    def end_recording(self):
        self.recording_active = False
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
        if self.writer is not None:
            self.writer.stop()
            self.writer = None

    def poll(self, decode_full=True, decode_preview=True, record=False):
        """Same shape as calibrate.CalibCameraSession.poll(), plus fps tracking
        (exponential smoothing on full-frame arrival interval, same pattern
        oak_camera.py/capture.py use).

        decode_full=False skips cv2.imdecode of the full-res JPEG (~16ms/camera
        at 4K, measured -- actually the LARGER of the two costs the ChArUco
        detection toggle was meant to address, and the one that toggle alone
        didn't touch: it only skipped detect_charuco, not this decode, which
        ran unconditionally regardless of the toggle). fps is still tracked
        from message arrival either way -- same "skip the decode, not the
        timing" pattern oak_camera.py's fps-only mode already established.
        last_frame_full is left at its previous value when skipped (stale,
        not cleared) since callers that need it check decode_full themselves
        before reading it.

        decode_preview=False skips decoding/exposing the preview frame (and
        the exposure-metadata read that comes with it) -- used by capture
        mode's preview auto-throttle to shed GUI-thumbnail cost under CPU
        pressure without touching the recording path. self.preview_updated
        reports whether the preview frame actually refreshed this call, so
        callers can skip a redundant GUI push when throttled.

        record=True (only meaningful once recording_active) writes the
        full-res packet straight to the take's BackgroundVideoWriter + a
        per-frame log line and returns early -- decode_full is NOT consulted
        in that case, since the packet bytes were already consumed for
        recording; last_frame_full is left stale for that tick (same
        "freeze, don't disappear" precedent the ChArUco-detection toggle
        already established for calibrate mode).
        """
        self.preview_updated = False
        preview_msg = self.q_preview.tryGet()
        if preview_msg is not None and decode_preview:
            self.last_frame_preview = preview_msg.getCvFrame()
            # Actual applied exposure (as opposed to what was last *requested*
            # via set_exposure) -- ImgFrame reports the real per-frame value,
            # confirmed live to update within a handful of frames of a
            # set_exposure() call. Used to give the exposure modal real
            # confirmation feedback instead of a fire-and-forget send.
            self.actual_shutter_us = round(preview_msg.getExposureTime().total_seconds() * 1_000_000)
            self.actual_iso = preview_msg.getSensitivity()
            self.preview_updated = True

        full_msg = self.q_full.tryGet()
        if full_msg is None:
            return False

        self.last_frame_full_ts = time.time()
        now_mono = time.monotonic()
        if self._fps_prev_mono is not None:
            inst = 1.0 / max(now_mono - self._fps_prev_mono, 1e-6)
            self.fps = inst if self.fps is None else self.fps * 0.8 + inst * 0.2
        self._fps_prev_mono = now_mono

        if record and self.recording_active:
            data = calibrate._packet_bytes(full_msg)
            self.writer.write_packet(data)
            ts = full_msg.getTimestamp()
            device_ts_s = ts.total_seconds() if ts is not None else float("nan")
            self.log_file.write(
                f"{self.last_frame_full_ts:.9f} {full_msg.getSequenceNum()} "
                f"{device_ts_s:.9f} {len(data)}\n"
            )
            return True

        if not decode_full:
            return True

        frame = cv2.imdecode(
            np.frombuffer(calibrate._packet_bytes(full_msg), dtype=np.uint8), cv2.IMREAD_COLOR,
        )
        if frame is None:
            return False
        self.last_frame_full = frame
        return True

    def reconnect(self):
        """Only one pipeline shape exists (see start_calibration_pipeline),
        so a mid-take reconnect comes back on the same stream a take is
        recording from -- recording_active/writer/log_file are untouched by
        this, so recording resumes seamlessly once frames start flowing
        again via poll(record=True).
        """
        available = dai.Device.getAllAvailableDevices()
        match = next((d for d in available if d.deviceId == self.device_id), None)
        if match is None:
            return False
        self.device_info = match
        self.connect()
        self.start_calibration_pipeline()
        return True

    def close(self):
        self.connected = False
        if self.recording_active:
            try:
                self.end_recording()
            except Exception:
                pass
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        if self.device is not None:
            try:
                self.device.close()
            except Exception:
                pass


# ==============================================================================
# Boot: connect every discovered camera once, with live per-device status in
# a modal. Order is deterministic -- see camera_boot_stats.rank_boot_order --
# not random, with a stagger between connects (same rationale as capture.py's
# main() for the stagger; see module docstring above for the ordering).
# ==============================================================================
def boot_cameras(viser_mgr, cfg):
    device_infos = dai.Device.getAllAvailableDevices()
    if not device_infos:
        raise RuntimeError("No OAK devices discovered.")
    device_infos = camera_boot_stats.rank_boot_order(device_infos)

    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sessions = {}
    with viser_mgr.server.gui.add_modal("Booting OAK cameras...") as modal:
        viser_mgr.server.gui.add_markdown(f"Found {len(device_infos)} device(s). Cameras with a "
                                           f"worse track record boot first (see camera_boot_stats.py).")
        status_rows = {
            info.deviceId: viser_mgr.server.gui.add_markdown(f"`{info.deviceId}` -- waiting...")
            for info in device_infos
        }

        CONNECT_STAGGER_S = 1.5
        for i, info in enumerate(device_infos):
            if i > 0:
                time.sleep(CONNECT_STAGGER_S)
            status_rows[info.deviceId].content = f"`{info.deviceId}` -- connecting..."
            cam_id = f"cam{i}"
            sess = AppCameraSession(cam_id, info, session_ts, i, cfg)
            try:
                sess.connect()
                sess.start_calibration_pipeline()
                sessions[cam_id] = sess
                status_rows[info.deviceId].content = (
                    f"`{info.deviceId}` -- **{cam_id}**, USB {sess.usb_speed.name}"
                )
            except Exception as exc:
                status_rows[info.deviceId].content = f"`{info.deviceId}` -- **FAILED**: {exc}"
                print(f"[Error] {cam_id} ({info.deviceId}) failed to boot: {exc}")

        close_btn = viser_mgr.server.gui.add_button("Continue")
        done = threading.Event()
        close_btn.on_click(lambda _: done.set())
        # Wait for the operator to actually click through -- important when
        # there's a failure to read, and `with modal:` itself does NOT close
        # the modal on exit (it only stops routing new GUI elements into it),
        # so without an explicit close() below the modal would stay on screen
        # forever regardless of this wait.
        done.wait()
    modal.close()
    return sessions


# ==============================================================================
# Calibration picker: load / start new / continue uncalibrated.
# ==============================================================================
def show_calibration_picker(viser_mgr, cfg, cmd_queue, sessions, cam_ids, intrinsics_cache, image_size,
                             alignment_detector, alignment_board_points_3d):
    """intrinsics_cache/image_size let this show, per booted camera, whether
    its intrinsics are already cached AND resolution-matched -- the exact
    same condition make_calib_state checks before treating a camera as
    intrinsics-locked, so what's shown here always matches what a
    "Start new calibration" click would actually do with that camera.

    Saved calibrations render one row per file, matching-camera files first
    (newest-first within each group) -- a file whose camera set doesn't
    exactly match the currently-booted rig gets disabled Load buttons
    rather than being hidden, so it's still visible but can't be picked.
    Loops (rather than returning immediately) so "Load & start capture"'s
    floor-board pre-flight check (verify_floor_board) can send the operator
    back to this same picker on Cancel instead of aborting the app.
    """
    calib_files = list_available_calibrations(cfg)
    active_device_ids = {sessions[cam_id].device_id for cam_id in cam_ids}
    match_info = {path: matches_active_cameras(path, active_device_ids) for path in calib_files}
    ordered_files = (
        [p for p in calib_files if match_info[p][0]] + [p for p in calib_files if not match_info[p][0]]
    )

    while True:
        with viser_mgr.server.gui.add_modal("Calibration") as modal:
            camera_lines = []
            for cam_id in cam_ids:
                session = sessions[cam_id]
                entry = intrinsics_cache.get(session.device_id)
                if entry is None:
                    status = "_no cached intrinsics_"
                elif entry["image_width"] != image_size[0] or entry["image_height"] != image_size[1]:
                    status = (f"_cached intrinsics are for {entry['image_width']}x{entry['image_height']}, "
                              f"this run is {image_size[0]}x{image_size[1]} -- will recalibrate_")
                else:
                    status = f"**intrinsics cached** ({entry['reprojection_error_px']:.3f}px reproj)"
                camera_lines.append(
                    f"- **{cam_id}** (`{session.device_id}`, USB {session.usb_speed.name}): {status}"
                )
            viser_mgr.server.gui.add_markdown("**Connected cameras:**\n\n" + "\n".join(camera_lines))

            if ordered_files:
                for path in ordered_files:
                    matches, cam_count = match_info[path]
                    label = os.path.basename(path)
                    note = "" if matches else " -- camera mismatch"
                    viser_mgr.server.gui.add_markdown(
                        f"**{label}** ({cam_count} camera{'s' if cam_count != 1 else ''}){note}"
                    )
                    load_btn = viser_mgr.server.gui.add_button("Load", disabled=not matches)
                    load_btn.on_click(lambda _, p=path: cmd_queue.put(("load_calibration", p)))
                    capture_btn = viser_mgr.server.gui.add_button("Load & start capture", disabled=not matches)
                    capture_btn.on_click(lambda _, p=path: cmd_queue.put(("load_and_capture", p)))
            else:
                viser_mgr.server.gui.add_markdown("_No saved calibrations found in "
                                                   f"`{cfg['output']['dir']}`._")
            start_btn = viser_mgr.server.gui.add_button("Start new calibration")
            start_btn.on_click(lambda _: cmd_queue.put(("start_calibration", None)))
            skip_btn = viser_mgr.server.gui.add_button("Continue uncalibrated")
            skip_btn.on_click(lambda _: cmd_queue.put(("uncalibrated", None)))

        choice = cmd_queue.get()  # blocks the setup thread until the operator picks one
        modal.close()

        if choice[0] == "load_and_capture":
            path = choice[1]
            loaded_candidate = calibrate.load_calibration_output(path)
            loaded_cameras_candidate = {
                entry["device_id"]: entry for entry in loaded_candidate.values() if entry.get("device_id")
            }
            proceed = verify_floor_board(
                viser_mgr, sessions, cam_ids, loaded_cameras_candidate,
                alignment_detector, alignment_board_points_3d, cfg,
            )
            if proceed:
                return choice
            continue  # Cancel -- re-show the picker

        return choice


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Combined calibration + capture app (viser)")
    parser.add_argument("--config", default=calibrate.CONFIG_PATH_DEFAULT,
                         help="Path to the YAML config file (shared with calibrate.py)")
    args = parser.parse_args()

    if dai is None:
        print("[Error] depthai is not installed in this environment.")
        return

    cfg = calibrate.load_config(args.config)
    if psutil is not None:
        psutil.cpu_percent(interval=None)  # prime -- first call has no baseline, see update below
    _board, detector, board_points_3d = calibrate.build_board(cfg)
    _alignment_board, alignment_detector, alignment_board_points_3d = calibrate.build_board(cfg, "alignment_board")
    image_size = (cfg["camera"]["record_width"], cfg["camera"]["record_height"])

    viser_mgr = calibrate.ViserManager(cfg)
    print(f"[Viser] http://localhost:{viser_mgr.server.get_port()}")

    try:
        sessions = boot_cameras(viser_mgr, cfg)
    except RuntimeError as exc:
        print(f"[Error] {exc}")
        return
    if not sessions:
        print("[Error] No cameras booted successfully -- nothing to do.")
        return
    cam_ids = list(sessions.keys())

    # Loaded here (rather than in the "Persistent state" block below) so the
    # calibration picker can show, per camera, whether its intrinsics are
    # already cached before the operator picks load/start-new/uncalibrated.
    intrinsics_cache = calibrate.load_intrinsics_cache(cfg) if cfg["intrinsics_cache"]["enabled"] else {}

    setup_queue = queue.Queue()
    action, action_arg = show_calibration_picker(
        viser_mgr, cfg, setup_queue, sessions, cam_ids, intrinsics_cache, image_size,
        alignment_detector, alignment_board_points_3d,
    )

    # ── Persistent state, shared by whichever mode is active ──────────────────
    calib_states = {cam_id: calibrate.CameraCalibState(cam_id, image_size, cfg) for cam_id in cam_ids}
    pose_graph = calibrate.PoseGraph(cfg)
    pose_result = None
    world_align = {"pending": False, "R": None, "t": None}
    # app_state["mode"]: "calibrate" (live engine running), "loaded" (static
    # poses from a saved file), or "uncalibrated" (no pose info at all).
    app_state = {"mode": "uncalibrated"}
    loaded_cameras = {}  # device_id -> {"K","dist","width","height","R","t"} when mode == "loaded"

    loaded_calibration_path = action_arg if action in ("load_calibration", "load_and_capture") else None

    if action in ("load_calibration", "load_and_capture"):
        loaded = calibrate.load_calibration_output(action_arg)
        loaded_cameras = {entry["device_id"]: entry for entry in loaded.values() if entry.get("device_id")}
        app_state["mode"] = "loaded"
        print(f"[Calibration] Loaded {action_arg} -- {len(loaded_cameras)} camera(s) with known pose.")
        for cam_id in cam_ids:
            entry = loaded_cameras.get(sessions[cam_id].device_id)
            if entry is None:
                print(f"[Calibration] {cam_id} ({sessions[cam_id].device_id}): not in loaded file -- pose unknown.")
    elif action == "start_calibration":
        app_state["mode"] = "calibrate"
        calib_states = {
            cam_id: calibrate.make_calib_state(
                cam_id, image_size, cfg, sessions[cam_id].device_id, intrinsics_cache,
            )
            for cam_id in cam_ids
        }
        print("[Calibration] Live calibration started.")
    else:
        print("[Calibration] Continuing uncalibrated.")

    # ── Persistent GUI chrome ───────────────────────────────────────────────
    # Live previews get their own standalone, dockable panel (viser.PanelHandle,
    # 1.1.0+) on the left, instead of living in the default sidebar with
    # everything else. _ensure_camera_panel only *creates* each camera's
    # folder/widgets the first time it's called (cam_id in self.cam_images
    # guards it) -- calling it once here, inside the panel's tab, is enough
    # to route those folders into this panel; later update_thumbnail*/
    # update_status calls just mutate the already-created handles and don't
    # care what container is "current" at that point.
    preview_panel = viser_mgr.server.gui.add_panel()
    with preview_panel.add_tab("Cameras"):
        for cam_id in cam_ids:
            viser_mgr._ensure_camera_panel(cam_id)
    preview_panel.dock_left()
    preview_panel.set_width(252)  # 70% of the previous 360

    viser_mgr.add_connection_banner()
    camera_settings_folder = viser_mgr.add_camera_settings_panel()  # static readout of the starting config values
    viser_mgr.add_performance_panel()

    # Exposure control lives in a modal (opened on demand) rather than an
    # always-visible slider pair -- a slider alone gives no feedback on
    # whether a change actually reached the cameras, so this tracks a pending
    # request and the main loop below confirms it against each camera's own
    # reported actual exposure (ImgFrame.getExposureTime()/getSensitivity(),
    # not just "the send didn't raise").
    exposure_state = {
        "shutter": cfg["camera"]["shutter_us"], "iso": cfg["camera"]["iso"],
        "pending": False, "status_md": None,
    }

    def open_exposure_modal():
        modal = viser_mgr.server.gui.add_modal("Exposure")
        with modal:
            viser_mgr.server.gui.add_markdown(
                "Live control, applied to every connected camera via CameraControl "
                "(not a config change) -- see AppCameraSession.set_exposure."
            )
            iso_slider = viser_mgr.server.gui.add_slider(
                "ISO", min=100, max=1600, step=50, initial_value=exposure_state["iso"],
            )
            shutter_slider = viser_mgr.server.gui.add_slider(
                "Shutter (us)", min=1, max=33000, step=100, initial_value=exposure_state["shutter"],
            )
            status_md = viser_mgr.server.gui.add_markdown(
                f"_Currently: shutter={exposure_state['shutter']}us, iso={exposure_state['iso']}_"
            )
            apply_btn = viser_mgr.server.gui.add_button("Apply")
            close_btn = viser_mgr.server.gui.add_button("Close")

        def _apply(_):
            exposure_state["shutter"] = shutter_slider.value
            exposure_state["iso"] = iso_slider.value
            exposure_state["pending"] = True
            exposure_state["status_md"] = status_md
            status_md.content = "_Applying..._"
            for session in sessions.values():
                session.set_exposure(shutter_slider.value, iso_slider.value)

        def _close(_):
            exposure_state["status_md"] = None  # about to be removed by modal.close()
            modal.close()

        apply_btn.on_click(_apply)
        close_btn.on_click(_close)

    with camera_settings_folder:
        exposure_btn = viser_mgr.server.gui.add_button("Exposure...")
        exposure_btn.on_click(lambda _: open_exposure_modal())

    # Global "Undistort views" toggle -- pure display flag, no device access,
    # so it's safe to flip directly from the callback thread (unlike
    # anything touching dai pipelines/queues). Applied in Pass 1 to every
    # camera's preview thumbnail via maybe_undistort, which gracefully
    # passes a frame through unchanged for any camera lacking real
    # intrinsics for the current mode.
    undistort_state = {"enabled": False}
    undistort_btn = viser_mgr.server.gui.add_button("Undistort views: OFF")

    def _toggle_undistort(_):
        undistort_state["enabled"] = not undistort_state["enabled"]
        undistort_btn.label = f"Undistort views: {'ON' if undistort_state['enabled'] else 'OFF'}"

    undistort_btn.on_click(_toggle_undistort)

    viser_mgr.add_view_controls()

    cmd_queue = queue.Queue()

    def request_world_alignment():
        if app_state["mode"] != "calibrate":
            print("[Align] World alignment is only available during a live calibration session "
                  "(click 'Start new calibration' first).")
            return
        world_align["pending"] = True
        print("[Align] Waiting for a fresh detection of the alignment board "
              "(any one camera) to set the world's origin and down direction...")

    # Lets the operator pause the expensive part of Pass 2 (both the full-res
    # JPEG decode and ChArUco detection -- see poll()'s decode_full param,
    # which this toggle also gates) without leaving calibrate mode entirely.
    detection_state = {"enabled": True}

    # ── Capture mode state -- see README.md's "Capture mode" section ────────
    capture_cmd_queue = queue.Queue()  # separate from cmd_queue: must work in any app_state["mode"]
    capture_state = {
        "phase": "idle",       # idle | starting | recording | stopping
        "session_ts": None,    # set on first "start" press, NOT boot time
        "take_n": 0,
        "take_name": None,
        "take_dir": None,
    }
    perf_state = {"cpu_pcts": [], "last_sample_mono": 0.0, "throttled": False}
    preview_mode = {"choice": "Auto"}  # "Auto" | "Always on" | "Always off"
    postproc = postprocess.PostprocessWorker()
    postproc.start()

    # add_tab_group() itself is a plain handle, not a context manager -- only
    # the individual add_tab(...) handles support `with`.
    tab_group = viser_mgr.server.gui.add_tab_group()
    with tab_group.add_tab("Calibrate"):
        viser_mgr.server.gui.add_markdown(
            f"Mode: **{app_state['mode']}**. Use Save/Reset/Uncache below during a live session."
        )
        detection_checkbox = viser_mgr.server.gui.add_checkbox(
            "ChArUco detection enabled", initial_value=True,
        )

        def _toggle_detection(_):
            detection_state["enabled"] = detection_checkbox.value
            if not detection_checkbox.value:
                # Hide rather than freeze -- the checkbox itself already
                # makes the paused state obvious, so a stale-looking board
                # pose left on screen would be misleading, not just old.
                for cam_id in cam_ids:
                    viser_mgr.hide_board_pose(cam_id)

        detection_checkbox.on_update(_toggle_detection)
        viser_mgr.add_world_alignment_button(request_world_alignment)
        viser_mgr.add_global_controls(cmd_queue)
    with tab_group.add_tab("Capture"):
        preview_mode_dropdown = viser_mgr.server.gui.add_dropdown(
            "Live previews", ("Auto", "Always on", "Always off"), initial_value="Auto",
        )
        preview_mode_dropdown.on_update(
            lambda _: preview_mode.__setitem__("choice", preview_mode_dropdown.value)
        )
        capture_status_md = viser_mgr.server.gui.add_markdown("_Idle._")
        capture_btn = viser_mgr.server.gui.add_button("Start capture")
        capture_btn.on_click(lambda _: capture_cmd_queue.put(
            "stop" if capture_state["phase"] == "recording" else "start"
        ))

    if action == "load_and_capture":
        # Skip the button click entirely -- "Load & start capture" already
        # got its confirmation via verify_floor_board's modal. Seeding the
        # phase here means the main loop's existing starting-phase handling
        # (_do_start_take, below) kicks off take_1 on its very first tick.
        capture_state["phase"] = "starting"
        capture_btn.label, capture_btn.disabled = "Starting...", True
        capture_status_md.content = "_Starting take..._"

    # Always-visible shutdown control -- styled as a clearly destructive
    # action (red + power icon) and placed last, so it renders at the
    # bottom of the panel. Enqueue-then-drain, same idiom as every other
    # button in this file: on_click only opens a confirm modal; only that
    # modal's "Confirm shutdown" button actually sets shutdown_event. The
    # main loop thread (see shutdown_event.is_set() check below) does the
    # actual, bounded cleanup and then force-exits the process -- see
    # main()'s tail-end comment for why a forced exit is used.
    shutdown_event = threading.Event()
    shutdown_btn = viser_mgr.server.gui.add_button("Shutdown app", color="red", icon="power")

    def _open_shutdown_modal():
        modal = viser_mgr.server.gui.add_modal("Confirm shutdown")
        with modal:
            viser_mgr.server.gui.add_markdown("Shut down the app? This closes every camera and exits.")
            confirm_btn = viser_mgr.server.gui.add_button("Confirm shutdown", color="red", icon="power")
            cancel_btn = viser_mgr.server.gui.add_button("Cancel")

        def _confirm(_):
            shutdown_btn.label, shutdown_btn.disabled = "Shutting down...", True
            shutdown_event.set()
            modal.close()

        confirm_btn.on_click(_confirm)
        cancel_btn.on_click(lambda _: modal.close())

    shutdown_btn.on_click(lambda _: _open_shutdown_modal())

    def clear_extrinsics_for(target):
        """Same as calibrate.py's own helper of this name."""
        nonlocal pose_graph, pose_result
        if target == "all":
            pose_graph = calibrate.PoseGraph(cfg)
            pose_result = None
            world_align["pending"] = False
            world_align["R"] = None
            world_align["t"] = None
            for cam_id in cam_ids:
                viser_mgr.remove_camera_pose(cam_id)
            viser_mgr.hide_alignment_board_pose("origin")
            print("[Reset] Pose graph and world alignment cleared.")
        elif target in calib_states:
            stale = [key for key in pose_graph.observations if target in key]
            for key in stale:
                del pose_graph.observations[key]
            viser_mgr.remove_camera_pose(target)
            if stale:
                print(f"[Reset] {target}: {len(stale)} pose-graph edge(s) cleared.")

    def write_take_meta(take_dir, cameras_meta):
        """Logs which calibration was active, or 'uncalibrated' if not --
        the two cases the request itself distinguishes; a take started
        mid live-calibration ("calibrate" mode, nothing saved/loaded yet)
        is logged as uncalibrated too, since no calibration is *loaded*.
        """
        if app_state["mode"] == "loaded":
            calibration_meta = {"mode": "loaded", "path": loaded_calibration_path}
        else:
            calibration_meta = {"mode": "uncalibrated", "path": None}
        meta = {
            "session_ts": capture_state["session_ts"],
            "take_n": capture_state["take_n"],
            "take_name": capture_state["take_name"],
            "start_time_iso": datetime.now().isoformat(),
            "start_time_unix": time.time(),
            "calibration": calibration_meta,
            "fps": cfg["camera"]["fps"],
            "resolution": [cfg["camera"]["record_width"], cfg["camera"]["record_height"]],
            "mjpeg_quality": cfg["camera"]["mjpeg_quality"],
            "cameras": cameras_meta,
        }
        with open(os.path.join(take_dir, "take_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return meta

    def _do_start_take():
        """Blocking (runs on the main loop thread, see module docstring's
        single-thread-dai-access invariant), but cheap: there is only ever
        one pipeline (see AppCameraSession.start_calibration_pipeline) --
        capture mode never swaps to a different one, so no camera is
        reconnected or rebooted when entering capture mode. Every take just
        opens a fresh writer/log per camera on the pipeline that's already
        running.
        """
        if capture_state["session_ts"] is None:
            capture_state["session_ts"] = datetime.now().strftime("%Y%m%d_%H%M%S")

        connected_sessions = [sessions[cam_id] for cam_id in cam_ids if sessions[cam_id].connected]

        capture_state["take_n"] += 1
        capture_state["take_name"] = f"take_{capture_state['take_n']}"
        take_dir = os.path.join(
            cfg["capture"]["dir"], capture_state["session_ts"], capture_state["take_name"],
        )
        os.makedirs(take_dir, exist_ok=True)
        capture_state["take_dir"] = take_dir

        if app_state["mode"] == "loaded":
            calibration_line = f"calibration_mode=loaded calibration_path={loaded_calibration_path}"
        else:
            calibration_line = "calibration_mode=uncalibrated"
        header_extra = [
            f"session_ts={capture_state['session_ts']} take_n={capture_state['take_n']} "
            f"take_name={capture_state['take_name']}",
            calibration_line,
        ]

        cameras_meta = {}
        for session in connected_sessions:
            try:
                session.capture_sync_calibration(take_dir)
            except Exception as exc:
                print(f"[Capture] {session.cam_id}: sync calibration failed: {exc}")
            video_path = os.path.join(take_dir, f"video_{session.cam_id}.mjpeg")
            log_path = os.path.join(take_dir, f"frame_timestamps_{session.cam_id}.log")
            session.begin_recording(video_path, log_path, header_extra)
            cameras_meta[session.cam_id] = {
                "device_id": session.device_id,
                "usb_speed": session.usb_speed.name if session.usb_speed else None,
                "shutter_us": session.actual_shutter_us,
                "iso": session.actual_iso,
                "video_path": video_path,
                "log_path": log_path,
                "sync_host_offset_s": session.sync_host_offset_s,
            }

        write_take_meta(take_dir, cameras_meta)
        print(f"[Capture] {capture_state['take_name']} recording started -- "
              f"{len(cameras_meta)} camera(s).")

    def _do_stop_take():
        """Blocking (BackgroundVideoWriter.stop() joins its writer thread,
        draining any queued frames -- a plain Python thread, safe to block
        the main loop on briefly). Enqueues postprocessing but does NOT wait
        for it -- the next take is free to start immediately.
        """
        for cam_id in cam_ids:
            session = sessions[cam_id]
            if session.recording_active:
                session.end_recording()
        if capture_state["take_dir"] is not None:
            postproc.queue.put(capture_state["take_dir"])
        print(f"[Capture] {capture_state['take_name']} stopped -- queued for postprocessing.")

    runtime_cfg = cfg["runtime"]
    tol_s = cfg["extrinsics"]["simultaneity_tolerance_ms"] / 1000.0
    last_status_print = time.monotonic()
    last_reconnect_attempt = {cam_id: 0.0 for cam_id in cam_ids}

    try:
        while True:
            if shutdown_event.is_set():
                print("[Stop] Shutdown requested from the UI.")
                break

            if app_state["mode"] != "calibrate":
                # Commands below (save/reset/uncache) all assume a live session;
                # in loaded/uncalibrated mode just drain and ignore/report them.
                while not cmd_queue.empty():
                    cmd = cmd_queue.get()
                    print(f"[Command] '{cmd}' ignored -- not in a live calibration session.")
            else:
                while not cmd_queue.empty():
                    cmd = cmd_queue.get()
                    if cmd == "save":
                        device_ids = {cam_id: sessions[cam_id].device_id for cam_id in cam_ids}
                        calibrate.save_output(cam_ids, calib_states, pose_result, cfg, device_ids)
                    elif cmd.startswith("reset"):
                        parts = cmd.split()
                        target = parts[1] if len(parts) > 1 else "all"
                        for cam_id in (cam_ids if target == "all" else [target]):
                            if cam_id in calib_states:
                                calib_states[cam_id] = calibrate.make_calib_state(
                                    cam_id, image_size, cfg, sessions[cam_id].device_id, intrinsics_cache,
                                )
                                print(f"[Reset] {cam_id} calibration state cleared.")
                        clear_extrinsics_for(target)
                    elif cmd.startswith("uncache"):
                        parts = cmd.split()
                        target = parts[1] if len(parts) > 1 else "all"
                        targets = cam_ids if target == "all" else [target]
                        evicted = False
                        for cam_id in targets:
                            if cam_id not in sessions:
                                continue
                            device_id = sessions[cam_id].device_id
                            evicted_entry = intrinsics_cache.pop(device_id, None)
                            if evicted_entry is not None:
                                evicted = True
                                backup_path = calibrate.backup_intrinsics_cache_entry(cfg, device_id, evicted_entry)
                                print(f"[Cache] {cam_id}: evicted cache entry for device {device_id} "
                                      f"(backed up to {backup_path}).")
                            if cam_id in calib_states:
                                calib_states[cam_id] = calibrate.CameraCalibState(cam_id, image_size, cfg)
                                print(f"[Reset] {cam_id} calibration state cleared (intrinsics uncached).")
                        if evicted and cfg["intrinsics_cache"]["enabled"]:
                            calibrate._atomic_write_json(cfg["intrinsics_cache"]["path"], intrinsics_cache)
                        clear_extrinsics_for(target)
                    else:
                        print(f"[Command] Unrecognized: {cmd!r}")

            # Capture button: enqueue-then-drain, same idiom as cmd_queue/
            # exposure_state above -- the on_click callback (a viser
            # background thread) only ever queues, this loop thread does the
            # actual (blocking) pipeline/recording work. GUI feedback
            # ("Starting..."/"Stopping...") is pushed on this same tick,
            # immediately before that blocking call, so the browser shows it
            # before the wait rather than after.
            while not capture_cmd_queue.empty():
                cmd = capture_cmd_queue.get()
                if cmd == "start" and capture_state["phase"] == "idle":
                    capture_state["phase"] = "starting"
                    capture_btn.label, capture_btn.disabled = "Starting...", True
                    capture_status_md.content = "_Starting take..._"
                elif cmd == "stop" and capture_state["phase"] == "recording":
                    capture_state["phase"] = "stopping"
                    capture_btn.label, capture_btn.disabled = "Stopping...", True
                    capture_status_md.content = "_Stopping..._"

            if capture_state["phase"] == "starting":
                _do_start_take()
                capture_state["phase"] = "recording"
                capture_btn.label, capture_btn.disabled = "Stop", False
            elif capture_state["phase"] == "stopping":
                _do_stop_take()
                capture_state["phase"] = "idle"
                capture_btn.label, capture_btn.disabled = "Start capture", False
                capture_status_md.content = "_Idle._"

            if capture_state["phase"] == "recording":
                dot = "\U0001F534" if (time.time() % 1.0) < 0.5 else "⚪"
                capture_status_md.content = f"{dot} **Recording** -- {capture_state['take_name']}"

            now = time.monotonic()

            # CPU sampling (once/sec -- psutil.cpu_percent must only ever be
            # sampled from this one call site, see update_performance's
            # docstring) drives the Performance panel and the preview
            # auto-throttle below. percpu=True, NOT a system-wide average --
            # one core pegged at 100% (ffmpeg encode, GIL-bound Python work)
            # while the rest idle would average out to a modest, easy-to-miss
            # number on a multi-core box; the max per-core value is what
            # actually signals "something is bottlenecked", so that's what
            # drives the throttle. Hysteresis (separate throttle/release
            # thresholds) avoids flapping previews on/off right at one
            # boundary value.
            if psutil is not None and now - perf_state["last_sample_mono"] >= 1.0:
                perf_state["last_sample_mono"] = now
                perf_state["cpu_pcts"] = psutil.cpu_percent(interval=None, percpu=True)
                cam_fps = {cam_id: sessions[cam_id].fps for cam_id in cam_ids}
                viser_mgr.update_performance(perf_state["cpu_pcts"], postproc.queue.qsize(), cam_fps)

            max_cpu_pct = max(perf_state["cpu_pcts"]) if perf_state["cpu_pcts"] else 0.0
            if capture_state["phase"] == "recording":
                if perf_state["throttled"]:
                    if max_cpu_pct < runtime_cfg["cpu_throttle_release_pct"]:
                        perf_state["throttled"] = False
                elif max_cpu_pct >= runtime_cfg["cpu_throttle_threshold_pct"]:
                    perf_state["throttled"] = True
            else:
                perf_state["throttled"] = False

            if preview_mode["choice"] == "Always off":
                previews_on = False
            elif preview_mode["choice"] == "Always on":
                previews_on = True
            else:  # Auto -- operator's explicit choice above always wins over this
                previews_on = not perf_state["throttled"]

            # Pass 1: fetch/decode every camera's latest frame -- always runs,
            # regardless of mode, since live previews are always shown.
            fresh_frames = {}
            for cam_id in cam_ids:
                session = sessions[cam_id]
                if not session.connected:
                    calib_states[cam_id].connected = False
                    if now - last_reconnect_attempt[cam_id] >= runtime_cfg["reconnect_retry_interval_s"]:
                        last_reconnect_attempt[cam_id] = now
                        try:
                            if session.reconnect():
                                calib_states[cam_id].connected = True
                                print(f"[Reconnect] {cam_id} reconnected.")
                        except Exception as exc:
                            print(f"[Reconnect] {cam_id} failed: {exc}")
                    continue
                # Only decode the full-res frame when something will actually
                # use it this tick -- "loaded" mode's own sanity-check
                # detection always needs it; "calibrate" mode needs it only
                # while the ChArUco toggle is on; "uncalibrated" never needs
                # it at all. This is the real fix for the detection toggle not
                # moving fps much: detect_charuco was gated already, but the
                # (larger, ~16ms/camera at 4K) JPEG decode above it was not.
                # While actually recording this camera, the full-res packet
                # is consumed by the recording write path instead (see
                # poll()'s record= param) -- decode_full is moot there.
                recording_now = capture_state["phase"] == "recording" and session.recording_active
                need_full_decode = (
                    not recording_now
                    and (app_state["mode"] == "loaded"
                         or (app_state["mode"] == "calibrate" and detection_state["enabled"]))
                )
                try:
                    got_frame = session.poll(
                        decode_full=need_full_decode, decode_preview=previews_on, record=recording_now,
                    )
                except Exception as exc:
                    print(f"[Error] {cam_id} disconnected: {exc}")
                    session.close()
                    calib_states[cam_id].connected = False
                    continue

                if session.preview_updated and session.last_frame_preview is not None:
                    state = calib_states.get(cam_id)
                    preview_frame = session.last_frame_preview
                    if undistort_state["enabled"]:
                        undistort_K = undistort_dist = None
                        if app_state["mode"] == "loaded":
                            entry = loaded_cameras.get(session.device_id)
                            if entry is not None:
                                undistort_K, undistort_dist = entry["K"], entry["dist"]
                        elif (app_state["mode"] == "calibrate" and state is not None
                              and state.has_intrinsics_estimate):
                            undistort_K, undistort_dist = state.K, state.dist
                        preview_frame = maybe_undistort(
                            preview_frame, undistort_K, undistort_dist,
                            (cfg["camera"]["display_width"], cfg["camera"]["display_height"]),
                            (cfg["camera"]["record_width"], cfg["camera"]["record_height"]),
                        )
                    if app_state["mode"] == "calibrate" and state is not None and not state.intrinsics_locked:
                        viser_mgr.update_thumbnail(cam_id, preview_frame, state.coverage)
                    else:
                        viser_mgr.update_thumbnail_raw(cam_id, preview_frame)

                if need_full_decode and got_frame and session.last_frame_full is not None:
                    fresh_frames[cam_id] = (session.last_frame_full, session.last_frame_full_ts)

            if exposure_state["pending"]:
                confirmed = all(
                    s.actual_shutter_us == exposure_state["shutter"] and s.actual_iso == exposure_state["iso"]
                    for s in sessions.values() if s.connected
                )
                if confirmed:
                    exposure_state["pending"] = False
                    if exposure_state["status_md"] is not None:
                        exposure_state["status_md"].content = (
                            f"_Applied to {len(sessions)} camera(s): "
                            f"shutter={exposure_state['shutter']}us, iso={exposure_state['iso']}_ ✓"
                        )

            # Pass 2: mode-dependent. Skipped entirely while detection_state
            # is disabled -- frustums/board poses just freeze at their last
            # value rather than disappearing (see detection_checkbox above).
            # Timed + counted (last_pass2_ms/last_pass2_cams) so the periodic
            # status print can show ground truth about what's actually
            # running, rather than relying on the fps number alone to infer it.
            pass2_t0 = time.perf_counter()
            last_pass2_cams = len(fresh_frames)
            if app_state["mode"] == "calibrate" and detection_state["enabled"]:
                for cam_id, (frame, frame_ts) in fresh_frames.items():
                    state = calib_states[cam_id]
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    if world_align["pending"]:
                        align_detection = calibrate.detect_charuco(alignment_detector, gray)
                        align_pose = None
                        if align_detection is not None:
                            a_corners2d, a_ids = align_detection
                            if len(a_ids) >= cfg["quality_gates"]["min_corners"]:
                                align_pose = calibrate.solve_board_pose(
                                    alignment_board_points_3d[a_ids], a_corners2d, state.K, state.dist,
                                )
                        if align_pose is not None:
                            a_R, a_t, _a_err = align_pose
                            state.last_alignment_pose = (frame_ts, a_R, a_t)
                            if pose_result is not None and cam_id in pose_result["poses"]:
                                T_cam_to_world = calibrate.invert_T(pose_result["poses"][cam_id])
                                viser_mgr.update_alignment_board_pose(
                                    cam_id, T_cam_to_world @ calibrate.rt_to_T(a_R, a_t),
                                )
                            else:
                                viser_mgr.hide_alignment_board_pose(cam_id)
                        else:
                            viser_mgr.hide_alignment_board_pose(cam_id)

                    detection = calibrate.detect_charuco(detector, gray)
                    if detection is None:
                        viser_mgr.hide_board_pose(cam_id)
                        continue
                    corners2d, ids = detection

                    pose, accepted = calibrate.process_detection(state, corners2d, ids, board_points_3d, gray, cfg)
                    if accepted:
                        was_converged = state.converged
                        calibrate.maybe_recalibrate(state, cfg)
                        if cfg["intrinsics_cache"]["enabled"] and state.converged and not was_converged:
                            calibrate.upsert_intrinsics_cache(cfg, sessions[cam_id].device_id, state, intrinsics_cache)
                    if pose is None:
                        viser_mgr.hide_board_pose(cam_id)
                        continue
                    R, t, err = pose
                    state.last_detection_ts = frame_ts
                    state.last_edge_pose = (frame_ts, R, t, err)
                    state.last_warned_no_detection = False

                    if pose_result is not None and cam_id in pose_result["poses"]:
                        T_cam_to_world = calibrate.invert_T(pose_result["poses"][cam_id])
                        viser_mgr.update_board_pose(cam_id, T_cam_to_world @ calibrate.rt_to_T(R, t))
                    else:
                        viser_mgr.hide_board_pose(cam_id)

                for i, cam_a in enumerate(cam_ids):
                    edge_a = calib_states[cam_a].last_edge_pose
                    if edge_a is None:
                        continue
                    ts_a, R_a, t_a, err_a = edge_a
                    for cam_b in cam_ids[i + 1:]:
                        edge_b = calib_states[cam_b].last_edge_pose
                        if edge_b is None:
                            continue
                        ts_b, R_b, t_b, err_b = edge_b
                        if abs(ts_a - ts_b) <= tol_s:
                            pose_graph.add_observation(cam_a, R_a, t_a, err_a, cam_b, R_b, t_b, err_b)

                if len(cam_ids) >= 2:
                    pose_result = pose_graph.solve(cam_ids)

                    if world_align["pending"]:
                        now_wall = time.time()
                        for cam_id in cam_ids:
                            align_pose = calib_states[cam_id].last_alignment_pose
                            if align_pose is None or cam_id not in pose_result["poses"]:
                                continue
                            pose_ts, R_bc, t_bc = align_pose
                            if now_wall - pose_ts > 2.0:
                                continue
                            T_cam_to_world = calibrate.invert_T(pose_result["poses"][cam_id])
                            world_align["R"], world_align["t"] = calibrate.compute_board_world_alignment(
                                T_cam_to_world, R_bc, t_bc,
                            )
                            world_align["pending"] = False
                            for c in cam_ids:
                                viser_mgr.hide_alignment_board_pose(c)
                            viser_mgr.update_alignment_board_pose(
                                "origin", calibrate.rt_to_T(calibrate._BOARD_UP_CORRECTION, np.zeros(3)),
                            )
                            viser_mgr.show_floor()
                            viser_mgr.set_view(position=(2.0, -6.0, 5.0), look_at=(0.0, 0.0, 0.0))
                            print(f"[Align] World origin/down direction set from {cam_id}'s "
                                  f"alignment-board detection.")
                            break
                    calibrate.apply_world_alignment(pose_result, world_align["R"], world_align["t"])

                    for cam_id in cam_ids:
                        state = calib_states[cam_id]
                        if cam_id in pose_result["poses"]:
                            T_cam_to_world = calibrate.invert_T(pose_result["poses"][cam_id])
                            viser_mgr.update_camera_pose(cam_id, T_cam_to_world, state.K, *state.image_size)
                        else:
                            viser_mgr.mark_pose_unknown(cam_id)

            elif app_state["mode"] == "loaded":
                # Read-only sanity check: poses are the fixed ones from the loaded
                # file (never re-solved), but each camera still detects the main
                # board live and shows it in 3D using those static intrinsics/pose
                # -- if the board doesn't line up as it's moved, the calibration
                # has likely gone stale and a fresh one should be run instead.
                for cam_id in cam_ids:
                    session = sessions[cam_id]
                    entry = loaded_cameras.get(session.device_id)
                    if entry is None:
                        viser_mgr.mark_pose_unknown(cam_id)
                        continue
                    T_cam_to_world = calibrate.invert_T(calibrate.rt_to_T(entry["R"], entry["t"]))
                    viser_mgr.update_camera_pose(cam_id, T_cam_to_world, entry["K"],
                                                  entry["width"], entry["height"])
                    if cam_id not in fresh_frames:
                        continue
                    frame, _frame_ts = fresh_frames[cam_id]
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    detection = calibrate.detect_charuco(detector, gray)
                    if detection is None:
                        viser_mgr.hide_board_pose(cam_id)
                        continue
                    corners2d, ids = detection
                    obj = board_points_3d[ids]
                    if len(ids) < cfg["quality_gates"]["min_corners"]:
                        viser_mgr.hide_board_pose(cam_id)
                        continue
                    solved = calibrate.solve_board_pose(obj, corners2d, entry["K"], entry["dist"])
                    if solved is None:
                        viser_mgr.hide_board_pose(cam_id)
                        continue
                    R, t, _err = solved
                    viser_mgr.update_board_pose(cam_id, T_cam_to_world @ calibrate.rt_to_T(R, t))

            else:  # uncalibrated
                for cam_id in cam_ids:
                    viser_mgr.mark_pose_unknown(cam_id)

            last_pass2_ms = (time.perf_counter() - pass2_t0) * 1000

            # Status panel: always shown, every mode -- fps is meaningful
            # regardless of calibration state; the calibration-fitting-specific
            # lines only make sense while a live session is actually running.
            for cam_id in cam_ids:
                session = sessions[cam_id]
                state = calib_states[cam_id]
                extra = []
                if app_state["mode"] == "calibrate":
                    if state.intrinsics_locked:
                        extra.append(f"_intrinsics: loaded from cache "
                                     f"({state.reproj_error:.3f}px at save time)_")
                    if state.last_detection_ts is not None:
                        silent_for = time.time() - state.last_detection_ts
                        if silent_for > runtime_cfg["no_detection_warning_s"]:
                            extra.append(f"**no detection for {silent_for:.0f}s**")
                    elif state.connected:
                        extra.append("**never detected the board yet**")
                viser_mgr.update_status(cam_id, state, pose_graph.sample_count_for(cam_id), extra)

            viser_mgr.update_connection_banner([cam_id for cam_id in cam_ids if not sessions[cam_id].connected])

            if now - last_status_print >= runtime_cfg["status_print_interval_s"]:
                last_status_print = now
                connected = sum(1 for s in sessions.values() if s.connected)
                det_str = "on" if detection_state["enabled"] else "off"
                top2_cpu = sorted(enumerate(perf_state["cpu_pcts"]), key=lambda pair: pair[1], reverse=True)[:2]
                cpu_str = ", ".join(f"{i}={p:.0f}%" for i, p in top2_cpu) if top2_cpu else "--"
                print(f"[Status] mode={app_state['mode']} | detection={det_str} | "
                      f"{connected}/{len(cam_ids)} camera(s) connected | "
                      f"pass2: {last_pass2_cams} cam(s) in {last_pass2_ms:.0f}ms | "
                      f"capture={capture_state['phase']} cpu top: {cpu_str} "
                      f"postprocess_queue={postproc.queue.qsize()}")

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n[Stop] Interrupted by user.")

    finally:
        capture.close_recorders_parallel(list(sessions.values()))
        if app_state["mode"] == "calibrate":
            device_ids = {cam_id: sessions[cam_id].device_id for cam_id in cam_ids}
            calibrate.save_output(cam_ids, calib_states, pose_result, cfg, device_ids)
        # Bounded -- ffmpeg over several 4K MJPEGs can legitimately take a
        # couple seconds, but exit must not hang waiting on it. Short on
        # purpose (was 60s): the whole point of the shutdown button/Ctrl-C
        # fix below is a *fast* exit, so a still-running job past this point
        # is abandoned (its take's raw footage is already safely on disk --
        # only that job's own postprocessing output is skipped) rather than
        # waited on.
        postproc.queue.put(None)
        postproc.join(timeout=8.0)
        if postproc.is_alive():
            print("[Stop] Postprocessing didn't finish in time -- abandoning it "
                  "(raw footage is unaffected; only its processed/ output is incomplete).")

    # Forced exit, not a normal return: found (via an isolated repro, no
    # app.py code involved) that ANY task submitted to viser's internal
    # ThreadPoolExecutor that never returns permanently blocks CPython's own
    # atexit machinery from letting the interpreter shut down, regardless of
    # daemon flags -- so a plain return here is not actually guaranteed to
    # exit the process. Everything that must run before exiting (closing
    # devices/files, saving calibration output, postprocessing) already
    # happened above with its own bounded timeout, so it's safe to force it
    # here rather than depend on every background thread (viser's, or
    # depthai's own internal ones) cooperatively joining.
    print("[Stop] Shutdown complete.")
    os._exit(0)


if __name__ == "__main__":
    main()
