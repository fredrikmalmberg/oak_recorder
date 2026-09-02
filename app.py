"""Combined calibration + capture UI (viser). Boots every discovered OAK
camera exactly once (reusing capture.py's hardened connect_device_with_retry
-- stagger, retries, boot-log JSONL -- for the same reason capture.py needed
it: OAK cameras draw a current spike while booting, and several connecting
close together can brown out whichever one boots last), then lets the
operator load a saved calibration, run a live one, or continue uncalibrated
-- all without re-booting cameras between those choices.

Boot is NOT one blocking pre-loop: sessions (one AppCameraSession per
discovered device, all connected=False) are created up front, and the
calibration picker is shown immediately -- it needs nothing but each
device's ID (already known from dai.Device.getAllAvailableDevices(), no
connection required). Actually connecting each camera (the slow,
staggered part) happens as a small step_boot() call, one camera at a
time, interleaved into whatever loop is currently running: the picker's
own wait loop while the operator is deciding, then the main tick loop
once it starts. This is deliberately NOT a background thread: see
"Single-thread dai-access rule" below.

Boot order is NOT random: camera_boot_stats.rank_boot_order sorts known
troublesome cameras (by past failure count, from logs/camera_boot_log.jsonl)
to the FRONT of the queue, since historical data showed failures cluster
almost entirely at the LAST boot position -- a camera with a bad track
record should never be the one left competing against every other
already-connected camera for power.

Single-thread dai-access rule: every dai/depthai call, for every camera,
for this app's entire life, happens on main()'s one thread -- step_boot,
session.poll(), session.reconnect(), all of it -- never a background
thread. This isn't arbitrary caution: Luxonis's own guidance (their
forum, "depthai-core thread safety specifically on dai::Device") is that
a Device object "isn't fully thread-safe," and the pattern they actually
recommend for multiple devices is one thread PER DEVICE for that
device's whole life, never handing a Device off between threads. This
app already satisfies a stricter version of that (one thread, period,
for every device) -- session.reconnect() already proves a brand-new
dai.Device connection works fine when it happens inline on this thread,
interleaved with poll() calls on other sessions; step_boot() is the same
pattern applied to first connect. Do not move boot (or anything else
that touches a session) onto a separate thread.

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
multi-camera boot/brownout risk step_boot's stagger exists to harden
against -- reproduced live (the same historically-bad camera
failed to reconnect) before this design was simplified to avoid it
entirely. So entering capture mode is instant: a take just starts writing
the already-running stream to disk. See README.md's "Capture mode" section
for the full session/take/postprocessing design.

The UI is two docked panels, both docked left so they land side by side.
Which one ends up at the screen edge is decided by PANEL CREATION order,
not by which dock_left() call happens first (a wrong fix tried once
already) -- viser's client applies every panel's placement in
Object.keys(panels) order (client/src/ControlPanel/placementCoordinator.
tsx), i.e. add_panel()-call order, and dockToEdge always inserts the panel
being processed as the new OUTERMOST column (client/src/dock/
layoutOps.test.ts -- the opposite of what PanelHandle.dock_left's own
Python docstring claims). So preview_panel is add_panel()'ed (and, to be
robust either way, also dock_left()'ed) FIRST, main_panel SECOND (see
main()), landing main_panel at the screen edge and preview_panel pushed
inward beside it. The outer one, "main" (300px wide), holds every button, split into
Calibration/Capture/Viewer content -- NOT viser tabs (viser exposes no way
to detect or set which tab is active from Python, GuiTabHandle/
GuiTabGroupHandle have neither an on-select callback nor an observable
"active" state, confirmed via introspection), but three add_folder(None)
groups inside one plain-header tab, toggled via .visible by three plain
buttons on the right-side default panel (mode_buttons, ui_mode_state,
_set_ui_mode) -- Python fully owns which mode is selected, unlike a tab
click. The inner panel, "live view" (960px, sized for its grid), holds
nothing but live camera imagery/take-video and each camera's metrics, in
one of three views, also driven by ui_mode_state via _set_ui_mode: a
2-column HTML grid of every camera's thumbnail (base64 JPEG data URIs,
that camera's status lines baked in as text underneath -- native GUI
widgets only stack in one column with no grid/row layout, confirmed via
introspection, so a raw HTML block is the only way to get a real grid,
and per-camera markdown can't be interleaved between its cells) during
Calibration; a single native image widget showing just the "middle"/
selected camera (highlighted green among the 3D view's frustums, see
wire_frustum_click and the main loop's per-tick highlight block) plus its
own metrics below it as markdown during Capture; or a selected past
take's grid video, played back the same way (see
_start_grid_video_playback/_select_viewer_take), during Viewer.
Per-camera Undistort checkboxes (plus an "Undistort all
cameras" checkbox that sets every one of them at once, one-shot -- it
doesn't stay synced if a per-camera checkbox is later toggled
individually) live in the live-view panel, next to the grid, since
they're a per-camera view option rather than a button; per-camera Reset
buttons stay with the rest of the buttons in main.
"""
import argparse
import base64
import html
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

import align_session
import calibrate
import camera_boot_stats
import capture
import postprocess

CALIBRATION_FILENAME_RE = re.compile(r"^\d{8}_\d{6}_\d+cam\.json$")
SESSION_DIR_RE = re.compile(r"^\d{8}_\d{6}$")
TAKE_DIR_RE = re.compile(r"^take_(\d+)$")


def list_sessions(cfg):
    """recordings/<session_ts>/ dirs, newest first -- same shape as
    list_available_calibrations below, just against the capture dir.
    """
    root = cfg["capture"]["dir"]
    if not os.path.isdir(root):
        return []
    names = [
        name for name in os.listdir(root)
        if SESSION_DIR_RE.match(name) and os.path.isdir(os.path.join(root, name))
    ]
    return sorted(names, reverse=True)


def session_display_name(cfg, session_ts):
    """session_meta.json's session_name if the operator typed one in the
    calibration picker (see show_calibration_picker/_do_start_take), else
    just the session_ts itself.
    """
    meta_path = os.path.join(cfg["capture"]["dir"], session_ts, "session_meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                name = json.load(f).get("session_name")
            if name:
                return name
        except (OSError, json.JSONDecodeError):
            pass
    return session_ts


def list_takes(session_dir):
    """take_<n> subfolders with a take_meta.json (skips a take directory
    that was created but never actually written, e.g. an interrupted
    boot), sorted by take number.
    """
    if not os.path.isdir(session_dir):
        return []
    found = []
    for name in os.listdir(session_dir):
        m = TAKE_DIR_RE.match(name)
        if m and os.path.isfile(os.path.join(session_dir, name, "take_meta.json")):
            found.append((int(m.group(1)), name))
    return [name for _n, name in sorted(found)]


def take_length_s(take_meta):
    """No duration/frame_count is ever persisted to take_meta.json (see
    write_take_meta) -- reconstruct it the same way align_session.py
    itself would: one camera's own frame_timestamps log, divided by fps.
    """
    first_cam = next(iter(take_meta["cameras"].values()))
    entries = align_session.parse_timestamp_log(first_cam["log_path"])
    return len(entries) / take_meta["fps"]


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


def make_preview_off_frame(width, height):
    """Dark placeholder with a crossed-out X, pushed into
    capture_preview_image whenever the "Live previews" toggle is off --
    reads as a deliberate blank rather than a stale frozen frame (Pass 1
    stops decoding/pushing entirely while the toggle is off, so nothing
    else would refresh the image on its own). Same idea as
    align_session.make_blue_frame's solid placeholder for a missing
    aligned frame, just with a cross drawn on top since this is an
    operator choice, not a missing-data condition.
    """
    img = np.full((height, width, 3), 40, dtype=np.uint8)
    thickness = max(2, width // 200)
    cv2.line(img, (0, 0), (width, height), (140, 60, 60), thickness=thickness)
    cv2.line(img, (width, 0), (0, height), (140, 60, 60), thickness=thickness)
    return img


def pick_middle_camera(cam_positions):
    """cam_positions: {cam_id: np.ndarray[3]} for cameras with a currently
    known pose (however rough -- works with as few as one, matching "as
    soon as we have the pose somewhat"). Cameras are arranged in a
    semicircle; the "middle" one is found by projecting each camera's
    (x, y) position onto the group's own dominant spread axis (PCA's first
    component) and taking the median along it.

    Deliberately NOT "angle around the centroid": for a semicircle (not a
    full circle), the point cloud's centroid sits well inside the arc, not
    at its true center, which biases angle-based ordering -- confirmed via
    a synthetic 5-camera semicircle test where it picked the wrong (2nd,
    not 3rd/true-middle) camera. Projecting onto the dominant spread axis
    doesn't depend on knowing the arc's actual center at all.
    """
    if not cam_positions:
        return None
    items = list(cam_positions.items())
    if len(items) == 1:
        return items[0][0]
    positions_xy = np.array([p[:2] for _, p in items])
    centered = positions_xy - positions_xy.mean(axis=0)
    _u, _s, vt = np.linalg.svd(centered)
    primary_axis = vt[0]
    projections = centered @ primary_axis
    median_idx = np.argsort(projections)[len(projections) // 2]
    return items[median_idx][0]


def build_camera_grid_html(cam_ids, latest_preview_frames, metrics=None, connected=None):
    """2-column CSS grid of base64-embedded JPEG thumbnails, each camera's
    status lines baked in as plain text underneath -- the only way to get
    both a real multi-column layout AND per-camera-positioned text out of
    viser (its native GUI widgets only stack in a single column with no
    grid/row/column container, confirmed via introspection, so a native
    markdown widget can't be interleaved between grid cells). latest_
    preview_frames holds whatever Pass 1 last decoded for each camera
    (already undistorted per-camera if that toggle is on, already
    coverage-overlaid in calibrate mode). metrics is an optional
    {cam_id: [line, ...]} of plain-text status lines -- HTML-escaped here
    so callers don't have to. connected is an optional {cam_id: bool} --
    a frameless cell shows "booting..." instead of "no signal" for any
    cam_id explicitly marked not-yet-connected there, so boot progress
    (see step_boot) reads as ongoing rather than as a dead camera.
    """
    metrics = metrics or {}
    connected = connected or {}
    cells = []
    for cam_id in cam_ids:
        frame = latest_preview_frames.get(cam_id)
        if frame is None:
            label = "booting..." if not connected.get(cam_id, True) else "no signal"
            image_html = (
                '<div style="aspect-ratio:16/9;background:#333;color:#aaa;'
                f'display:flex;align-items:center;justify-content:center">{label}</div>'
            )
        else:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            image_html = (
                f'<img src="data:image/jpeg;base64,{base64.b64encode(buf).decode("ascii")}" '
                f'style="width:100%;display:block">'
                if ok else '<div style="color:#aaa">decode failed</div>'
            )
        metric_html = "".join(f"<div>{html.escape(line)}</div>" for line in metrics.get(cam_id, []))
        cells.append(
            f'<div><div style="text-align:center;font-size:12px">{cam_id}</div>'
            f'{image_html}'
            f'<div style="font-size:11px;color:#ccc">{metric_html}</div></div>'
        )
    return (
        '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:4px">'
        + "".join(cells) + "</div>"
    )


def maybe_undistort(frame, K, dist, display_size, record_size):
    """K=None means "no usable intrinsics for this camera right now" (not
    yet converged / not in the loaded file / uncalibrated mode) -- returns
    frame unchanged rather than raising, which is what lets the "Undistort
    all cameras" checkbox apply blindly across every camera regardless of
    whether each one actually has intrinsics yet (Pass 1 prints a one-time
    warning per camera in this situation instead).

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
# Calibration picker: load / start new / continue uncalibrated. Shown
# immediately (device IDs only, no connection needed) -- see main() for
# where sessions/cam_ids get built and step_boot for the actual staggered
# per-camera connect this picker's wait loop drives in the background.
# ==============================================================================
def show_calibration_picker(viser_mgr, cfg, cmd_queue, sessions, cam_ids, intrinsics_cache, image_size,
                             alignment_detector, alignment_board_points_3d, step_boot, drain_boot_failures,
                             all_booted, session_ts):
    """intrinsics_cache/image_size let this show, per booted camera, whether
    its intrinsics are already cached AND resolution-matched -- the exact
    same condition make_calib_state checks before treating a camera as
    intrinsics-locked, so what's shown here always matches what a
    "Start new calibration" click would actually do with that camera.

    Saved calibrations whose camera set exactly matches the currently-booted
    rig get a real row (Load / Load & start capture buttons); everything
    else is just named under a "Mismatched calibrations" list -- no buttons,
    since they can't be loaded anyway, so there's nothing to click there.
    Loops (rather than returning immediately) so "Load & start capture"'s
    floor-board pre-flight check (verify_floor_board) can send the operator
    back to this same picker on Cancel instead of aborting the app.

    Shown before any camera is necessarily connected -- every device's
    intrinsics-cache/mismatch status only needs device_id (known instantly),
    but "Load & start capture" needs a live camera for its floor-board
    check, so those buttons stay disabled until all_booted() (see
    step_boot/main()). The wait for a click is a short-timeout poll, not a
    hard block, specifically so step_boot() keeps making progress and the
    "Connected cameras" list keeps refreshing while the operator decides.
    """
    calib_files = list_available_calibrations(cfg)
    active_device_ids = {sessions[cam_id].device_id for cam_id in cam_ids}
    match_info = {path: matches_active_cameras(path, active_device_ids) for path in calib_files}
    matching_files = [p for p in calib_files if match_info[p][0]]
    mismatched_files = [p for p in calib_files if not match_info[p][0]]

    def _camera_status_line(cam_id):
        session = sessions[cam_id]
        if session.usb_speed is None:
            return f"- **{cam_id}** (`{session.device_id}`): _booting..._"
        entry = intrinsics_cache.get(session.device_id)
        if entry is None:
            status = "_no cached intrinsics_"
        elif entry["image_width"] != image_size[0] or entry["image_height"] != image_size[1]:
            status = (f"_cached intrinsics are for {entry['image_width']}x{entry['image_height']}, "
                      f"this run is {image_size[0]}x{image_size[1]} -- will recalibrate_")
        else:
            status = f"**intrinsics cached** ({entry['reprojection_error_px']:.3f}px reproj)"
        return f"- **{cam_id}** (`{session.device_id}`, USB {session.usb_speed.name}): {status}"

    session_name_value = session_ts

    while True:
        load_and_capture_buttons = []
        with viser_mgr.server.gui.add_modal("Calibration") as modal:
            with viser_mgr.server.gui.add_folder("Get started"):
                start_btn = viser_mgr.server.gui.add_button("Start new calibration", color="green")
                start_btn.on_click(lambda _: cmd_queue.put(("start_calibration", None)))
                skip_btn = viser_mgr.server.gui.add_button("Continue uncalibrated", color="red")
                skip_btn.on_click(lambda _: cmd_queue.put(("uncalibrated", None)))
            # Pre-filled with the timestamp format already used elsewhere as
            # an implicit session name -- editable, read once a choice is
            # made (see the return below). Display name only; the
            # recordings/<session_ts>/ folder itself keeps the timestamp
            # naming (see session_meta.json in _do_start_take).
            session_name_input = viser_mgr.server.gui.add_text(
                "Session name", initial_value=session_name_value,
            )
            viser_mgr.server.gui.add_divider()

            camera_status_md = viser_mgr.server.gui.add_markdown(
                "**Connected cameras:**\n\n" + "\n".join(_camera_status_line(c) for c in cam_ids)
            )

            if matching_files:
                for path in matching_files:
                    _matches, cam_count = match_info[path]
                    label = os.path.basename(path)
                    viser_mgr.server.gui.add_markdown(
                        f"**{label}** ({cam_count} camera{'s' if cam_count != 1 else ''})"
                    )
                    load_btn = viser_mgr.server.gui.add_button("Load")
                    load_btn.on_click(lambda _, p=path: cmd_queue.put(("load_calibration", p)))
                    capture_btn = viser_mgr.server.gui.add_button(
                        "Load & start capture", disabled=not all_booted(),
                    )
                    capture_btn.on_click(lambda _, p=path: cmd_queue.put(("load_and_capture", p)))
                    load_and_capture_buttons.append(capture_btn)
            elif not mismatched_files:
                viser_mgr.server.gui.add_markdown("_No saved calibrations found in "
                                                   f"`{cfg['output']['dir']}`._")

            if mismatched_files:
                lines = [
                    f"- {os.path.basename(p)} ({match_info[p][1]} camera{'s' if match_info[p][1] != 1 else ''})"
                    for p in mismatched_files
                ]
                viser_mgr.server.gui.add_markdown(
                    "_Mismatched calibrations (different camera set -- can't be loaded):_\n\n"
                    + "\n".join(lines)
                )

        choice = None
        while choice is None:
            step_boot()
            drain_boot_failures()
            camera_status_md.content = (
                "**Connected cameras:**\n\n" + "\n".join(_camera_status_line(c) for c in cam_ids)
            )
            booted = all_booted()
            for btn in load_and_capture_buttons:
                btn.disabled = not booted
            try:
                choice = cmd_queue.get(timeout=0.05)
            except queue.Empty:
                continue
        session_name_value = session_name_input.value
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
                return choice[0], choice[1], session_name_value
            continue  # Cancel -- re-show the picker

        return choice[0], choice[1], session_name_value


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

    # Device IDs only, no connection -- fast. sessions is pre-populated for
    # every DISCOVERED device (all connected=False), not just ones that end
    # up connecting successfully, so the calibration picker (and everything
    # downstream) can show/gate on "still booting" instead of a camera
    # simply not existing yet. See module docstring's "Single-thread
    # dai-access rule" for why boot happens as step_boot() below rather
    # than a background thread.
    device_infos = dai.Device.getAllAvailableDevices()
    if not device_infos:
        print("[Error] No OAK devices discovered.")
        return
    device_infos = camera_boot_stats.rank_boot_order(device_infos)
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sessions = {
        f"cam{i}": AppCameraSession(f"cam{i}", info, session_ts, i, cfg)
        for i, info in enumerate(device_infos)
    }
    cam_ids = list(sessions.keys())

    CONNECT_STAGGER_S = 1.5
    boot_state = {
        "next_index": 0,
        "last_attempt_mono": -CONNECT_STAGGER_S,  # so cam0 boots on the very first step_boot() call
        "attempted": set(),   # cam_ids that have had a first boot attempt (see Pass 1's reconnect guard)
        "failures": [],       # (cam_id, device_id, error_str), drained into modals by drain_boot_failures
    }

    def step_boot():
        """At most one camera's connect+pipeline-start per call, staggered
        by CONNECT_STAGGER_S -- same brownout-avoidance stagger this used
        to run as one blocking pre-loop (see module docstring), just spread
        across many small steps on this same thread instead. Called from
        both the calibration picker's wait loop and the main tick loop, so
        boot keeps progressing regardless of which screen is showing. A
        no-op once every camera has had its first attempt.
        """
        if boot_state["next_index"] >= len(cam_ids):
            return
        now = time.monotonic()
        if now - boot_state["last_attempt_mono"] < CONNECT_STAGGER_S:
            return
        cam_id = cam_ids[boot_state["next_index"]]
        sess = sessions[cam_id]
        boot_state["attempted"].add(cam_id)
        try:
            sess.connect()
            sess.start_calibration_pipeline()
            if sess.usb_speed not in (dai.UsbSpeed.SUPER, dai.UsbSpeed.SUPER_PLUS):
                boot_state["failures"].append(
                    (cam_id, sess.device_id, f"USB {sess.usb_speed.name} (expected SuperSpeed)")
                )
        except Exception as exc:
            boot_state["failures"].append((cam_id, sess.device_id, str(exc)))
            print(f"[Error] {cam_id} ({sess.device_id}) failed to boot: {exc}")
        boot_state["next_index"] += 1
        boot_state["last_attempt_mono"] = now

    def all_booted():
        return boot_state["next_index"] >= len(cam_ids)

    def drain_boot_failures():
        """One dismiss-only modal per failure -- non-blocking, doesn't stop
        boot or whatever screen is currently showing. Called from the same
        two places as step_boot.
        """
        while boot_state["failures"]:
            cam_id, device_id, error = boot_state["failures"].pop(0)
            with viser_mgr.server.gui.add_modal(f"{cam_id} failed to boot") as fail_modal:
                viser_mgr.server.gui.add_markdown(f"`{device_id}`\n\n{error}")
                ok_btn = viser_mgr.server.gui.add_button("OK")
            ok_btn.on_click(lambda _, m=fail_modal: m.close())

    # Loaded here (rather than in the "Persistent state" block below) so the
    # calibration picker can show, per camera, whether its intrinsics are
    # already cached before the operator picks load/start-new/uncalibrated.
    intrinsics_cache = calibrate.load_intrinsics_cache(cfg) if cfg["intrinsics_cache"]["enabled"] else {}

    setup_queue = queue.Queue()
    action, action_arg, session_name = show_calibration_picker(
        viser_mgr, cfg, setup_queue, sessions, cam_ids, intrinsics_cache, image_size,
        alignment_detector, alignment_board_points_3d, step_boot, drain_boot_failures,
        all_booted, session_ts,
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
    # cmd_queue is also used by the Calibration tab's Save/Uncache/per-camera
    # Reset buttons (built further down, alongside the Capture tab), so it's
    # created up front rather than closer to where those live.
    cmd_queue = queue.Queue()

    viser_mgr.add_connection_banner()

    # Mode switcher: three plain buttons, not add_button_group -- a button
    # group can't do per-option color/disable (GuiButtonGroupHandle.disabled
    # is group-wide and its setter asserts False), and the ask here is the
    # selected mode gray+unclickable, the other two green. Each on_click is
    # wired later (see _set_ui_mode, below the panels it needs to exist),
    # same "define now, wire later" pattern as _open_take_video_modal.
    ui_mode_state = {"current": "Calibration"}
    UI_MODES = ("Calibration", "Capture", "Viewer")
    with viser_mgr.server.gui.add_folder("Mode"):
        mode_buttons = {mode: viser_mgr.server.gui.add_button(mode) for mode in UI_MODES}

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

    viser_mgr.add_view_controls()

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
    # which this toggle also gates). detection_state["enabled"] is ONLY
    # about that detect/solve pipeline now -- it deliberately does NOT
    # decide what the live-preview panel shows (see show_grid in the main
    # loop for that, keyed off app_state["mode"]/capture_state["phase"]
    # instead). The two used to share one flag; that was a bug (the Capture
    # tab's old "sanity check" checkbox could force the grid open during
    # actual capture work), fixed by decoupling them. Capture mode has no
    # detection control of its own now (see _set_ui_mode forcing this off
    # on entry) -- only the Calibration folder's checkbox reads/writes this
    # flag; see _set_detection.
    detection_state = {"enabled": True}

    # Per-camera undistort, set all at once via the "Undistort all cameras"
    # checkbox in the live-preview panel (there's no per-camera control
    # anymore -- see _set_undistort_all). Absent key == off, same as
    # capture_preview below.
    undistort_state = {}

    # Cameras that currently have undistort enabled but no usable
    # intrinsics yet (maybe_undistort is silently passing the raw frame
    # through) -- surfaced as a per-camera metrics line (see
    # per_cam_metrics in the main loop) instead of a console print.
    # Rebuilt fresh every tick in Pass 1, so this needs no throttling the
    # way a print-based warning would.
    undistort_missing_intrinsics = set()

    # wired_frustum_clicks tracks which camera frustums already have their
    # on_click handler attached (frustum handles are created once and
    # reused across pose updates -- attaching twice would double-fire a
    # click).
    wired_frustum_clicks = set()

    # Capture mode's single live preview: which camera, and whether that
    # choice is still auto-picked (pick_middle_camera, re-run every tick
    # pose data is available) or a manual override from clicking a frustum
    # in the 3D view (see Pass 2 below).
    capture_preview = {"selected": None, "auto": True}

    # Populated by Pass 1 each tick: cam_id -> latest (possibly undistorted/
    # coverage-overlaid) preview frame. Read by both the live-preview
    # panel's grid and its single Capture-view image, so neither has to
    # re-decode or duplicate that per-camera logic.
    latest_preview_frames = {}

    grid_state = {"last_update_mono": 0.0}

    mode_cmd_queue = queue.Queue()  # separate from cmd_queue: must work in any app_state["mode"]

    # ── Capture mode state -- see README.md's "Capture mode" section ────────
    capture_cmd_queue = queue.Queue()  # separate from cmd_queue: must work in any app_state["mode"]
    capture_state = {
        "phase": "idle",       # idle | starting | recording | stopping
        "session_ts": None,    # set on first "start" press, NOT boot time
        "session_name": session_name,  # typed at boot; written into session_meta.json on first take
        "take_n": 0,
        "take_name": None,
        "take_dir": None,
        # Every take this session, oldest first -- {take_n, take_name,
        # take_dir, start_time_unix, stop_time_unix (None while recording)}.
        # Unlike the fields above (which only ever hold the CURRENT/latest
        # take, overwritten in place by _do_start_take), this accumulates,
        # so the Capture tab's takes list (see the main loop) can show every
        # take's length and postprocessing status, not just the latest.
        "takes": [],
    }
    perf_state = {"cpu_pcts": [], "last_sample_mono": 0.0, "throttled": False}
    # Operator toggle (Capture mode's live-view panel only): off forces the
    # single-camera preview blank/crossed-out regardless of anything else;
    # on still respects the CPU auto-throttle below (perf_state["throttled"])
    # during an actual recording, same safety behavior the old "Auto" dropdown
    # choice had -- this toggle only removes the old "Always on" option,
    # which bypassed that throttle entirely.
    preview_toggle_state = {"on": True}
    postproc = postprocess.PostprocessWorker()
    postproc.start()

    # Panel ORDER here (which is created first) is what actually determines
    # left-to-right position, NOT the order dock_left() is called (that was
    # last turn's wrong fix). Confirmed by reading viser's client source:
    # the placement coordinator applies every panel's position in
    # Object.keys(panels) order -- i.e. panel-creation/registration order --
    # regardless of when each one's dock_left() command was issued
    # (client/src/ControlPanel/placementCoordinator.tsx: "for (const uuid of
    # Object.keys(panels)) processPanel(uuid)"), and dockToEdge always
    # inserts the panel being processed as the NEW OUTERMOST column
    # (client/src/dock/layoutOps.test.ts: "docks to the far-left (outermost)
    # when the edge already has content" -- the opposite of what
    # PanelHandle.dock_left's own Python docstring claims). So whichever
    # panel is CREATED (add_panel()'ed) second ends up outermost (at the
    # screen edge), pushing the first-created one inward. To land main_panel
    # at the screen edge with preview_panel to its right, preview_panel must
    # be created (and, to be robust to any arrival-order-vs-registration-
    # order subtlety, also docked) FIRST, main_panel SECOND.

    # Live-preview panel: imagery and per-camera metrics only, no buttons
    # (except the single "Undistort all cameras" checkbox -- a view option,
    # grouped here with the camera displays rather than with the rest of
    # the buttons in main_panel). Three mutually-exclusive content blocks
    # share this one tab (a panel always needs at least one add_tab() as a
    # content container -- see PanelHandle.__enter__ -- but a single tab
    # renders as a plain header, not a tab strip, so this isn't a visible
    # "tab" to the operator). Which block is visible is driven entirely by
    # ui_mode_state["current"] -- see _set_ui_mode, below.
    # 960x540, matching capture_frame's upscale below -- so the crossed-out
    # placeholder fills the panel the same way a real frame does, instead of
    # shrinking back down to the small native preview size.
    preview_off_frame = make_preview_off_frame(960, 540)

    preview_panel = viser_mgr.server.gui.add_panel()
    with preview_panel.add_tab("Live view"):
        grid_html = viser_mgr.server.gui.add_html(build_camera_grid_html(
            cam_ids, latest_preview_frames, connected={c: sessions[c].connected for c in cam_ids},
        ))
        preview_toggle_checkbox = viser_mgr.server.gui.add_checkbox(
            "Live previews", initial_value=preview_toggle_state["on"],
        )
        capture_preview_image = viser_mgr.server.gui.add_image(
            np.zeros((4, 4, 3), dtype=np.uint8), label="live",
        )
        capture_preview_status_md = viser_mgr.server.gui.add_markdown("_waiting for data..._")
        undistort_all_checkbox = viser_mgr.server.gui.add_checkbox(
            "Undistort all cameras", initial_value=False,
        )
        viewer_info_md = viser_mgr.server.gui.add_markdown("_No take selected._")
        viewer_video_image = viser_mgr.server.gui.add_image(np.zeros((4, 4, 3), dtype=np.uint8))
        viewer_video_status_md = viser_mgr.server.gui.add_markdown("", visible=False)

    def _set_undistort_all(enabled):
        """undistort_all_checkbox is the only operator control now
        (per-camera checkboxes removed) -- fans a single toggle out to
        every camera's undistort_state entry. Cameras without usable
        intrinsics yet just silently keep showing their raw frame (see
        maybe_undistort's K=None passthrough and the per-camera metrics
        warning in Pass 1 below).
        """
        for cam_id in cam_ids:
            undistort_state[cam_id] = enabled

    undistort_all_checkbox.on_update(lambda _: _set_undistort_all(undistort_all_checkbox.value))

    def _set_preview_toggle(enabled):
        """Off forces capture_preview_image to a deliberate blank/crossed-out
        placeholder immediately -- Pass 1 stops decoding/pushing entirely
        while off (see previews_on in the main loop), so without this the
        image would otherwise just freeze on whatever frame it last had.
        """
        preview_toggle_state["on"] = enabled
        if not enabled:
            capture_preview_image.image = preview_off_frame
            capture_preview_status_md.content = ""

    preview_toggle_checkbox.on_update(lambda _: _set_preview_toggle(preview_toggle_checkbox.value))
    preview_panel.dock_left()
    preview_panel.set_width(960)  # 2 wide columns, room for each camera's baked-in metrics text

    # Main/leftmost panel: buttons only, nothing else. A panel always needs
    # at least one add_tab() as its content container (PanelHandle.__enter__
    # raises otherwise), so this keeps exactly one, neutrally labeled --
    # a single-tab panel renders as a plain header, not a tab strip. The
    # three modes' content lives inside as three add_folder(None) groups
    # (label=None -> no header/border, pure layout grouping), each toggled
    # via .visible by _set_ui_mode (below, once every widget referenced
    # there exists) instead of being separate tabs.
    main_panel = viser_mgr.server.gui.add_panel()
    with main_panel.add_tab("Control"):
        with viser_mgr.server.gui.add_folder(None) as calibration_folder:
            mode_md = viser_mgr.server.gui.add_markdown(f"Mode: **{app_state['mode']}**")

            calibration_detection_checkbox = viser_mgr.server.gui.add_checkbox(
                "ChArUco detection enabled", initial_value=detection_state["enabled"],
            )

            viser_mgr.add_world_alignment_button(request_world_alignment)

            save_button = viser_mgr.server.gui.add_button("Save now", color="green")
            save_button.on_click(lambda _: cmd_queue.put("save"))

            viser_mgr.server.gui.add_divider()

            for cam_id in cam_ids:
                reset_btn = viser_mgr.server.gui.add_button(f"Reset {cam_id}")
                reset_btn.on_click(lambda _, c=cam_id: cmd_queue.put(f"reset {c}"))

            viser_mgr.server.gui.add_divider()

            start_btn = viser_mgr.server.gui.add_button("Start new calibration", color="red")
            start_btn.on_click(lambda _: mode_cmd_queue.put("start_calibration_live"))
            uncache_button = viser_mgr.server.gui.add_button("Uncache all intrinsics", color="red")
            uncache_button.on_click(lambda _: cmd_queue.put("uncache all"))

        with viser_mgr.server.gui.add_folder(None) as capture_folder:
            capture_status_md = viser_mgr.server.gui.add_markdown("_Idle._")
            capture_btn = viser_mgr.server.gui.add_button("Start capture", color="green")
            capture_btn.on_click(lambda _: capture_cmd_queue.put(
                "stop" if capture_state["phase"] == "recording" else "start"
            ))
            viser_mgr.server.gui.add_divider()
            no_takes_md = viser_mgr.server.gui.add_markdown("_No takes yet._")

        # (take_dir, take_meta, label_md) per past take found on disk --
        # refreshed every main-loop tick alongside take_row_widgets below
        # (see "Rebuild each take's row"), same always-live idiom as the
        # rest of this panel: every mode's widgets stay live regardless of
        # which folder is currently .visible.
        viewer_take_rows = []
        with viser_mgr.server.gui.add_folder(None) as viewer_folder:
            sessions_on_disk = list_sessions(cfg)
            if not sessions_on_disk:
                viser_mgr.server.gui.add_markdown("_No past sessions found._")
            for session_ts in sessions_on_disk:
                session_dir = os.path.join(cfg["capture"]["dir"], session_ts)
                take_names = list_takes(session_dir)
                with viser_mgr.server.gui.add_folder(
                    session_display_name(cfg, session_ts), expand_by_default=False,
                ):
                    if not take_names:
                        viser_mgr.server.gui.add_markdown("_No takes in this session._")
                    for take_name in take_names:
                        take_dir = os.path.join(session_dir, take_name)
                        with open(os.path.join(take_dir, "take_meta.json"), encoding="utf-8") as f:
                            take_meta = json.load(f)
                        # Length can't change for a past take -- compute it once
                        # here (parses a frame_timestamps log) rather than every
                        # tick; only postprocess status (a small JSON read) is
                        # worth refreshing live, since a job can still be running.
                        try:
                            length_str = f"{take_length_s(take_meta):.0f}s"
                        except (OSError, KeyError, StopIteration, ZeroDivisionError):
                            length_str = "?s"
                        label_md = viser_mgr.server.gui.add_markdown("")
                        play_btn = viser_mgr.server.gui.add_button("Play grid video")
                        # Selects this take for the live-view panel's embedded
                        # player (see _select_viewer_take) rather than opening
                        # a modal -- distinct from the Capture folder's own
                        # per-take "Play video" button below, which still uses
                        # _open_take_video_modal for takes made this run.
                        play_btn.on_click(lambda _, td=take_dir, tn=take_meta["take_n"]:
                                           _select_viewer_take(td, tn))
                        viser_mgr.server.gui.add_markdown("_3D playback (SMPLX) -- coming soon_")
                        viewer_take_rows.append((take_dir, take_meta["take_n"], length_str, label_md))

    main_panel.dock_left()
    main_panel.set_width(300)

    # take_n -> {"label_md": GuiMarkdownHandle, "play_btn": GuiButtonHandle},
    # one pair per take, built by _build_take_row as each take starts (see
    # _do_start_take) and kept updated in place every main-loop tick (see
    # the "Rebuild each take's row" block below) rather than torn down and
    # rebuilt, since viser has no way to reorder/replace GUI children --
    # only ever append within capture_folder.
    take_row_widgets = {}

    def _start_grid_video_playback(video_path, image_handle, on_status, fallback_fps):
        """Background cv2.VideoCapture -> GuiImageHandle.image frame-push
        loop -- the mechanism the live-preview panel already uses for
        camera frames (see capture_preview_image.image = ... in the main
        loop below), reused here for playing back a recorded grid video
        too. A first version instead pointed a raw <video src=...> tag (via
        add_html) at a second static-file HTTP server; that never rendered
        anything in real use, and there's no way to open the browser's
        console/network tab from here to diagnose why. Loops forever until
        the returned Event is set. on_status(msg) is called with a string
        to show a status line, or None once real frames are playing (to
        hide it). Shared by _open_take_video_modal (Capture folder's own
        per-take modal player) and _select_viewer_take (Viewer folder's
        embedded live-view player) -- same mechanism, different
        destination widget.
        """
        stop_event = threading.Event()

        def _playback_loop():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                on_status(f"_Could not open `{video_path}`_")
                return
            fps = cap.get(cv2.CAP_PROP_FPS) or fallback_fps
            frame_period_s = 1.0 / fps
            on_status(None)
            try:
                while not stop_event.is_set():
                    ok, frame_bgr = cap.read()
                    if not ok:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop back to the start
                        continue
                    image_handle.image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    stop_event.wait(frame_period_s)
            finally:
                cap.release()

        threading.Thread(target=_playback_loop, daemon=True).start()
        return stop_event

    def _open_take_video_modal(take):
        """Play a postprocessed take's grid video in a modal -- see
        _start_grid_video_playback for the actual decode/frame-push
        mechanism this wraps.
        """
        grid_video_path = align_session.grid_mp4_path(os.path.join(take["take_dir"], "processed"))

        modal = viser_mgr.server.gui.add_modal(f"Take {take['take_n']} -- grid video")
        with modal:
            # viser's add_modal has no width control (its GuiModalMessage
            # carries only title/order), and Mantine's default modal renders
            # at a fixed ~407px regardless of content -- confirmed via
            # Playwright that even an !important CSS override of `width`
            # alone does nothing, because Mantine sizes the dialog through
            # flex-basis, not width, so width has no effect on a flex child
            # with a fixed flex-basis. Forcing all three (width, max-width,
            # flex-basis) via JS after the dialog exists is what actually
            # works. There's no direct Python->JS bridge (add_html sinks
            # through dangerouslySetInnerHTML, which never runs <script>
            # tags), so this reuses the same img-onerror trick as the
            # hostname lookup earlier in this file's history: image loads
            # are a real async network request, so onerror always fires,
            # regardless of how the <img> was inserted -- unlike the <svg
            # onload=...> tried first for that, which empirically never
            # fired for a programmatically-inserted inline SVG. The retry
            # loop matters: onerror can fire before Mantine's modal
            # transition has actually attached .mantine-Modal-content to
            # the DOM (confirmed via Playwright -- a bare, unretried lookup
            # intermittently found no ancestor and silently did nothing).
            # 1221px = 3x the measured ~407px default, per the user's ask;
            # max-width keeps it from overflowing a narrower browser window.
            viser_mgr.server.gui.add_html(
                '<img src="__resolve_modal_size__" style="display:none" onerror="'
                "const self = this; let tries = 0; (function widen() { "
                "const c = self.closest('.mantine-Modal-content'); "
                "if (c) { "
                "c.style.setProperty('width', '1221px', 'important'); "
                "c.style.setProperty('max-width', '90vw', 'important'); "
                "c.style.setProperty('flex-basis', '1221px', 'important'); "
                "self.remove(); "
                "} else if (tries++ < 20) { setTimeout(widen, 50); } "
                "else { self.remove(); }"
                "})();"
                '">'
            )
            video_image = viser_mgr.server.gui.add_image(np.zeros((4, 4, 3), dtype=np.uint8))
            status_md = viser_mgr.server.gui.add_markdown("_Loading..._")
            close_btn = viser_mgr.server.gui.add_button("Close")

        def _set_status(msg):
            status_md.visible = msg is not None
            status_md.content = msg or ""

        stop_event = _start_grid_video_playback(
            grid_video_path, video_image, _set_status, cfg["camera"]["fps"],
        )

        def _close(_):
            stop_event.set()
            modal.close()

        close_btn.on_click(_close)

    viewer_playback_state = {"stop_event": None}

    def _select_viewer_take(take_dir, take_n):
        """Selects a past take for the live-view panel's embedded player
        (see _set_ui_mode, which shows/hides viewer_video_image based on
        ui_mode_state) -- replaces the Viewer folder's old modal-based
        player. Stops any previous take's playback thread first.
        """
        if viewer_playback_state["stop_event"] is not None:
            viewer_playback_state["stop_event"].set()
        viewer_info_md.content = f"**Take {take_n}**\n\n`{take_dir}`"
        viewer_video_status_md.visible = True
        viewer_video_status_md.content = "_Loading..._"
        grid_video_path = align_session.grid_mp4_path(os.path.join(take_dir, "processed"))

        def _set_status(msg):
            viewer_video_status_md.visible = msg is not None
            viewer_video_status_md.content = msg or ""

        viewer_playback_state["stop_event"] = _start_grid_video_playback(
            grid_video_path, viewer_video_image, _set_status, cfg["camera"]["fps"],
        )

    def _build_take_row(take):
        no_takes_md.visible = False
        with capture_folder:
            label_md = viser_mgr.server.gui.add_markdown("")
            play_btn = viser_mgr.server.gui.add_button("Play video", visible=False)
        play_btn.on_click(lambda _: _open_take_video_modal(take))
        take_row_widgets[take["take_n"]] = {"label_md": label_md, "play_btn": play_btn}

    def _set_detection(enabled):
        """Sets detection_state and the Calibration folder's own checkbox
        (the only one now -- Capture mode has no detection control of its
        own, see _set_ui_mode forcing this off on entry) and centralizes
        the hide-board-poses-on-off behavior.
        """
        detection_state["enabled"] = enabled
        calibration_detection_checkbox.value = enabled
        if not enabled:
            for cam_id in cam_ids:
                viser_mgr.hide_board_pose(cam_id)

    calibration_detection_checkbox.on_update(lambda _: _set_detection(calibration_detection_checkbox.value))

    def _set_ui_mode(mode):
        """The single place every mode-dependent thing gets set -- which
        main_panel folder is visible, what the live-view panel shows, and
        (entering Capture) forcing detection off. No early-return guard for
        "already selected": the clicked button is disabled while selected
        so it can't re-fire, and the one intentional same-mode call (the
        initial _set_ui_mode("Calibration") below) is naturally idempotent.
        """
        ui_mode_state["current"] = mode
        for m, btn in mode_buttons.items():
            selected = m == mode
            btn.color = "gray" if selected else "green"
            btn.disabled = selected

        calibration_folder.visible = mode == "Calibration"
        capture_folder.visible = mode == "Capture"
        viewer_folder.visible = mode == "Viewer"

        grid_html.visible = mode == "Calibration"
        preview_toggle_checkbox.visible = mode == "Capture"
        capture_preview_image.visible = mode == "Capture"
        capture_preview_status_md.visible = mode == "Capture"
        undistort_all_checkbox.visible = mode != "Viewer"
        viewer_info_md.visible = mode == "Viewer"
        viewer_video_image.visible = mode == "Viewer"
        viewer_video_status_md.visible = mode == "Viewer" and viewer_video_status_md.content != ""

        if mode == "Capture" and not preview_toggle_state["on"]:
            # Show the blank/crossed-out placeholder right away rather than
            # whatever frame happened to be there from before -- Pass 1
            # won't push anything new while the toggle is off.
            capture_preview_image.image = preview_off_frame

        if mode != "Viewer" and viewer_playback_state["stop_event"] is not None:
            # Stop decoding while the panel is hidden; the take stays
            # "remembered" (viewer_info_md keeps its content) but playback
            # doesn't auto-resume on re-entry -- the operator clicks the
            # take again if they want to keep watching.
            viewer_playback_state["stop_event"].set()
            viewer_playback_state["stop_event"] = None

        if mode == "Capture":
            _set_detection(False)

    for _mode, _btn in mode_buttons.items():
        _btn.on_click(lambda _, m=_mode: _set_ui_mode(m))
    _set_ui_mode("Calibration")  # initial state: matches "always start in Calibration"

    if action == "load_and_capture":
        # Skip the button click entirely -- "Load & start capture" already
        # got its confirmation via verify_floor_board's modal. Seeding the
        # phase here means the main loop's existing starting-phase handling
        # (_do_start_take, below) kicks off take_1 on its very first tick.
        capture_state["phase"] = "starting"
        capture_btn.label, capture_btn.disabled, capture_btn.color = "Starting...", True, "red"
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
            # remove_camera_pose destroys the frustum handle -- the next
            # pose update creates a fresh one, which needs its on_click
            # re-wired (see Pass 2's frustum-click-wiring block).
            wired_frustum_clicks.clear()
            print("[Reset] Pose graph and world alignment cleared.")
        elif target in calib_states:
            stale = [key for key in pose_graph.observations if target in key]
            for key in stale:
                del pose_graph.observations[key]
            viser_mgr.remove_camera_pose(target)
            if stale:
                print(f"[Reset] {target}: {len(stale)} pose-graph edge(s) cleared.")

    def wire_frustum_click(cam_id):
        """Attaches the click-to-select-for-Capture-preview handler to a
        camera's frustum, once -- frustum handles are created once and
        reused across pose updates (calibrate.py's update_camera_pose only
        creates a new one if cam_id isn't already in viser_mgr.frustums), so
        calling on_click twice would double-fire a single click. Coloring
        itself happens centrally, every tick (see the main loop's frustum
        highlight block below), not here -- so the highlight stays correct
        for an auto-picked selection too, not just a manual click.
        """
        if cam_id in wired_frustum_clicks:
            return
        frustum = viser_mgr.frustums.get(cam_id)
        if frustum is None:
            return
        wired_frustum_clicks.add(cam_id)

        def _on_click(_):
            capture_preview["selected"] = cam_id
            capture_preview["auto"] = False

        frustum.on_click(_on_click)

    def start_new_calibration_live():
        """Same as picking 'Start new calibration' at boot (main()'s own
        action == "start_calibration" branch) -- exposed live too, since
        without this there was no way back into "calibrate" mode once the
        operator had left it, which also left world alignment permanently
        refusing (it requires app_state["mode"] == "calibrate"). Doubles as
        the reset-out-of-"loaded" path: it clears loaded_cameras/
        loaded_calibration_path/pose graph/frustums the same way a plain
        reset would, so there's no separate "reset to uncalibrated" control.
        """
        nonlocal loaded_calibration_path
        app_state["mode"] = "calibrate"
        loaded_cameras.clear()
        loaded_calibration_path = None
        for cam_id in cam_ids:
            calib_states[cam_id] = calibrate.make_calib_state(
                cam_id, image_size, cfg, sessions[cam_id].device_id, intrinsics_cache,
            )
            viser_mgr.hide_board_pose(cam_id)
        clear_extrinsics_for("all")
        mode_md.content = "Mode: **calibrate**"
        print("[Calibration] Live calibration (re)started.")

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
        "Single-thread dai-access rule"), but cheap: there is only ever
        one pipeline (see AppCameraSession.start_calibration_pipeline) --
        capture mode never swaps to a different one, so no camera is
        reconnected or rebooted when entering capture mode. Every take just
        opens a fresh writer/log per camera on the pipeline that's already
        running.
        """
        if capture_state["session_ts"] is None:
            capture_state["session_ts"] = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = os.path.join(cfg["capture"]["dir"], capture_state["session_ts"])
            os.makedirs(session_dir, exist_ok=True)
            with open(os.path.join(session_dir, "session_meta.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "session_ts": capture_state["session_ts"],
                    "session_name": capture_state["session_name"],
                }, f, indent=2)

        connected_sessions = [sessions[cam_id] for cam_id in cam_ids if sessions[cam_id].connected]

        capture_state["take_n"] += 1
        capture_state["take_name"] = f"take_{capture_state['take_n']}"
        take_dir = os.path.join(
            cfg["capture"]["dir"], capture_state["session_ts"], capture_state["take_name"],
        )
        os.makedirs(take_dir, exist_ok=True)
        capture_state["take_dir"] = take_dir
        new_take = {
            "take_n": capture_state["take_n"],
            "take_name": capture_state["take_name"],
            "take_dir": take_dir,
            "start_time_unix": time.time(),
            "stop_time_unix": None,
        }
        capture_state["takes"].append(new_take)
        _build_take_row(new_take)

        if app_state["mode"] == "loaded":
            calibration_line = f"calibration_mode=loaded calibration_path={loaded_calibration_path}"
        else:
            calibration_line = "calibration_mode=uncalibrated"
        header_extra = [
            f"session_ts={capture_state['session_ts']} take_n={capture_state['take_n']} "
            f"take_name={capture_state['take_name']}",
            calibration_line,
            # One-time snapshot of what the operator had the live-preview
            # panel set to at the moment this take started -- NOT re-sampled
            # during recording, so this only answers "what was going on
            # when the take began," not a timeline of changes mid-take.
            # detection_enabled logs pre_capture_enabled (the operator's
            # setting before the mutual-exclusion logic force-disabled it
            # for the take, set just above in the "starting"-phase handler)
            # rather than detection_state["enabled"] itself, which is
            # always False during a take by construction and would be
            # useless for correlating detection with capture timing.
            f"preview_toggle={'on' if preview_toggle_state['on'] else 'off'} ui_mode={ui_mode_state['current']} "
            f"detection_enabled={detection_state.get('pre_capture_enabled', False)} "
            f"preview_camera={capture_preview['selected']}",
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
        if capture_state["takes"]:
            capture_state["takes"][-1]["stop_time_unix"] = time.time()
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

            while not mode_cmd_queue.empty():
                cmd = mode_cmd_queue.get()
                if cmd == "start_calibration_live":
                    start_new_calibration_live()

            # Capture button: enqueue-then-drain, same idiom as cmd_queue/
            # exposure_state above -- the on_click callback (a viser
            # background thread) only ever queues, this loop thread does the
            # actual (blocking) pipeline/recording work. GUI feedback
            # ("Starting..."/"Stopping...") is pushed on this same tick,
            # immediately before that blocking call, so the browser shows it
            # before the wait rather than after.
            while not capture_cmd_queue.empty():
                cmd = capture_cmd_queue.get()
                if cmd == "start" and capture_state["phase"] == "idle" and not detection_state["enabled"]:
                    capture_state["phase"] = "starting"
                    capture_btn.label, capture_btn.disabled, capture_btn.color = "Starting...", True, "red"
                    capture_status_md.content = "_Starting take..._"
                elif cmd == "stop" and capture_state["phase"] == "recording":
                    capture_state["phase"] = "stopping"
                    capture_btn.label, capture_btn.disabled, capture_btn.color = "Stopping...", True, "green"
                    capture_status_md.content = "_Stopping..._"

            if capture_state["phase"] == "starting":
                # ChArUco detection is expensive (~30ms/camera decode+detect,
                # see the fps investigation this session's history) and adds
                # nothing to a recording -- confirmed against a real 6-camera
                # take that leaving it on drags recording down to ~6fps.
                # Force it off for the duration of the take regardless of
                # the checkbox, remembering the operator's own prior choice
                # so stopping restores it rather than always re-enabling.
                detection_state["pre_capture_enabled"] = detection_state["enabled"]
                if detection_state["enabled"]:
                    _set_detection(False)
                _do_start_take()
                capture_state["phase"] = "recording"
                capture_btn.label, capture_btn.disabled, capture_btn.color = "Stop", False, "red"
            elif capture_state["phase"] == "stopping":
                _do_stop_take()
                capture_state["phase"] = "idle"
                capture_btn.label, capture_btn.disabled, capture_btn.color = "Start capture", False, "green"
                capture_status_md.content = "_Idle._"
                if detection_state.pop("pre_capture_enabled", False):
                    _set_detection(True)

            if capture_state["phase"] == "recording":
                dot = "\U0001F534" if (time.time() % 1.0) < 0.5 else "⚪"
                capture_status_md.content = f"{dot} **Recording** -- {capture_state['take_name']}"
            elif capture_state["phase"] == "idle":
                # Mutual exclusion: never let detection and recording run at
                # once (confirmed via a real take that leaving detection on
                # tanks recording fps ~5x) -- unlike the auto-off-during-a-
                # take above, this additionally blocks *starting* a take at
                # all while the operator has deliberately left detection on.
                capture_btn.disabled = detection_state["enabled"] or not all_booted()
                capture_status_md.content = (
                    "_Turn off ChArUco detection to capture._" if detection_state["enabled"]
                    else "_Waiting for cameras to finish booting..._" if not all_booted()
                    else "_Idle._"
                )

            # Every take this session, oldest first -- length, and whether
            # postprocessing has produced its grid MP4 yet (plus a Play
            # button once it has -- see _build_take_row/_open_take_video_modal
            # above). One markdown+button pair per take (take_row_widgets),
            # updated in place rather than rebuilding a single combined
            # widget, since a Play button needs its own click handler per
            # take. Cheap enough (a handful of takes, one small JSON read
            # each) to just refresh every tick rather than throttle. Status
            # comes from postprocess.read_status(), which reads the status
            # file PostprocessWorker writes as it works ("queued" ->
            # "processing" -> "done"/"failed") -- not inferred from whether
            # the output mp4 exists, so a failed job shows as failed instead
            # of stuck showing "processing...".
            for take in capture_state["takes"]:
                if take["stop_time_unix"] is not None:
                    length_str = f"{take['stop_time_unix'] - take['start_time_unix']:.0f}s"
                else:
                    length_str = f"{time.time() - take['start_time_unix']:.0f}s (recording)"
                status = postprocess.read_status(take["take_dir"])
                status_str = {
                    "queued": "queued for postprocessing",
                    "processing": "processing...",
                    "done": "postprocessed",
                    "failed": f"postprocessing failed ({status.get('error', 'unknown error')})",
                }.get(status["status"], status["status"])
                row = take_row_widgets[take["take_n"]]
                row["label_md"].content = f"Take {take['take_n']}: {length_str} -- {status_str}"
                row["play_btn"].visible = status["status"] == "done"

            # Same status refresh for the Viewer folder's past-session takes
            # -- length was already computed once at startup (see
            # viewer_folder above), only postprocess status can still
            # change live.
            for take_dir, take_n, length_str, label_md in viewer_take_rows:
                status = postprocess.read_status(take_dir)
                status_str = {
                    "queued": "queued for postprocessing",
                    "processing": "processing...",
                    "done": "postprocessed",
                    "failed": f"postprocessing failed ({status.get('error', 'unknown error')})",
                }.get(status["status"], status["status"])
                label_md.content = f"Take {take_n}: {length_str} -- {status_str}"

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

            if not preview_toggle_state["on"]:
                previews_on = False
            else:  # still respects the CPU auto-throttle during an actual recording
                previews_on = not perf_state["throttled"]

            # Which live-preview view is showing -- driven entirely by
            # ui_mode_state["current"] (see _set_ui_mode, which also owns
            # the actual .visible toggling for every live-view widget).
            # Only the two booleans needed for Pass 1's decode gating below
            # are recomputed here.
            show_grid = ui_mode_state["current"] == "Calibration"
            show_single = ui_mode_state["current"] == "Capture"

            # Keep making progress on any camera that hasn't had its first
            # boot attempt yet -- a no-op once all_booted() (see step_boot's
            # own docstring). Also true if the operator picked a picker
            # option before boot finished.
            step_boot()
            drain_boot_failures()

            # Pass 1: fetch/decode every camera's latest frame. decode_preview
            # is per-camera: every camera while the grid is showing (it
            # needs all of them), the single selected camera while Capture's
            # single-camera view is showing, none while Viewer mode is
            # showing a recorded take instead -- no reason to pay the decode
            # cost for cameras nothing is showing live.
            fresh_frames = {}
            for cam_id in cam_ids:
                session = sessions[cam_id]
                if not session.connected:
                    calib_states[cam_id].connected = False
                    if cam_id not in boot_state["attempted"]:
                        # Still waiting its turn in the staggered initial
                        # boot (step_boot above) -- the generic reconnect
                        # retry below is for a camera that WAS connected and
                        # dropped, not this one; retrying it here too would
                        # boot several cameras back-to-back in one tick and
                        # reintroduce the exact brownout risk the stagger
                        # exists to avoid.
                        continue
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
                # use it this tick -- both "loaded" mode's sanity-check
                # detection and "calibrate" mode's live detection are gated
                # on the same ChArUco toggle (detection_state["enabled"]);
                # "uncalibrated" never needs it at all. While actually
                # recording this camera, the full-res packet is consumed by
                # the recording write path instead (see poll()'s record=
                # param) -- decode_full is moot there, and
                # detection_state["enabled"] is itself forced off for the
                # duration of a take regardless of any checkbox (see the
                # "starting" phase transition below) specifically so this
                # can't silently keep running during a recording.
                recording_now = capture_state["phase"] == "recording" and session.recording_active
                need_full_decode = (
                    not recording_now
                    and detection_state["enabled"]
                    and app_state["mode"] in ("loaded", "calibrate")
                )
                decode_this_preview = previews_on and (
                    show_grid or (show_single and cam_id == capture_preview["selected"])
                )
                try:
                    got_frame = session.poll(
                        decode_full=need_full_decode, decode_preview=decode_this_preview, record=recording_now,
                    )
                except Exception as exc:
                    print(f"[Error] {cam_id} disconnected: {exc}")
                    session.close()
                    calib_states[cam_id].connected = False
                    continue

                if session.preview_updated and session.last_frame_preview is not None:
                    state = calib_states.get(cam_id)
                    preview_frame = session.last_frame_preview
                    if undistort_state.get(cam_id, False):
                        undistort_K = undistort_dist = None
                        if app_state["mode"] == "loaded":
                            entry = loaded_cameras.get(session.device_id)
                            if entry is not None:
                                undistort_K, undistort_dist = entry["K"], entry["dist"]
                        elif (app_state["mode"] == "calibrate" and state is not None
                              and state.has_intrinsics_estimate):
                            undistort_K, undistort_dist = state.K, state.dist
                        if undistort_K is None:
                            undistort_missing_intrinsics.add(cam_id)
                        else:
                            undistort_missing_intrinsics.discard(cam_id)
                        preview_frame = maybe_undistort(
                            preview_frame, undistort_K, undistort_dist,
                            (cfg["camera"]["display_width"], cfg["camera"]["display_height"]),
                            (cfg["camera"]["record_width"], cfg["camera"]["record_height"]),
                        )
                    else:
                        undistort_missing_intrinsics.discard(cam_id)  # cleared once undistort is off
                    if (detection_state["enabled"] and app_state["mode"] == "calibrate"
                            and state is not None and not state.intrinsics_locked):
                        preview_frame = calibrate.draw_coverage_overlay(preview_frame, state.coverage)
                    latest_preview_frames[cam_id] = preview_frame
                    if cam_id == capture_preview["selected"]:
                        # Pushed even while capture_preview_image.visible is
                        # False (detection currently on) so it's already
                        # fresh the instant the operator switches views.
                        # Upscaled to the live-view panel's own width (960,
                        # see preview_panel.set_width) before pushing --
                        # add_image's <img> only ever shrinks to fit its
                        # container (maxWidth: 100%, no width: 100%), so the
                        # raw 480x270 preview stream would render at its
                        # native (small) size instead of filling the panel
                        # now that it's the only thing showing in Capture
                        # mode. The camera's own preview stream stays small
                        # (see build_camera_pipeline's display_width/height)
                        # since the grid still needs 6 small thumbnails --
                        # this resize is local to the single-camera view.
                        capture_frame = cv2.resize(
                            preview_frame, (960, 540), interpolation=cv2.INTER_LINEAR,
                        )
                        capture_preview_image.image = cv2.cvtColor(capture_frame, cv2.COLOR_BGR2RGB)

                if need_full_decode and got_frame and session.last_frame_full is not None:
                    fresh_frames[cam_id] = (session.last_frame_full, session.last_frame_full_ts)

            if exposure_state["pending"]:
                confirmed = all(
                    s.actual_shutter_us == exposure_state["shutter"] and s.actual_iso == exposure_state["iso"]
                    for s in sessions.values() if s.connected
                )
                if confirmed:
                    exposure_state["pending"] = False
                    viser_mgr.update_camera_settings(exposure_state["shutter"], exposure_state["iso"])
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
                            wire_frustum_click(cam_id)
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
                    wire_frustum_click(cam_id)
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

            # Capture preview's "middle camera" -- re-picked every tick from
            # whatever poses currently exist (viser_mgr.frustums, populated
            # by whichever Pass 2 branch above ran) as long as the operator
            # hasn't manually clicked a different camera's frustum. Falls
            # back to auto if a manually-selected camera disconnects, rather
            # than showing a dead preview indefinitely.
            if not capture_preview["auto"] and capture_preview["selected"] is not None:
                sel_session = sessions.get(capture_preview["selected"])
                if sel_session is None or not sel_session.connected:
                    capture_preview["auto"] = True
            if capture_preview["auto"]:
                cam_positions = {
                    cam_id: np.array(viser_mgr.frustums[cam_id].position)
                    for cam_id in cam_ids if cam_id in viser_mgr.frustums
                }
                picked = pick_middle_camera(cam_positions)
                if picked is not None:
                    capture_preview["selected"] = picked

            # Highlight whichever camera is selected (auto-picked or
            # manually clicked, doesn't matter) green in the 3D view, every
            # tick -- not just at click time -- so a freshly (re)created
            # frustum (e.g. after a pose reset) still ends up the right
            # color on its very next appearance.
            for cam_id in cam_ids:
                frustum = viser_mgr.frustums.get(cam_id)
                if frustum is not None:
                    frustum.color = (
                        (0, 255, 0) if cam_id == capture_preview["selected"] else (60, 140, 220)
                    )

            # Per-camera metrics -- computed every tick (cheap string
            # formatting) since the grid wants up-to-date numbers whenever
            # it's actually visible. Plain text, not markdown: these lines
            # get baked as escaped HTML into the grid, so plain text is the
            # only formatting that renders correctly there. Coverage is
            # deliberately NOT repeated here as a percentage: the red/green
            # grid draw_coverage_overlay already paints it directly onto the
            # thumbnail (Pass 1 above). Capture mode's single-camera preview
            # deliberately does NOT reuse this -- it's all calibration debug
            # info (samples/reproj/converged), meaningless once a take is
            # actually being recorded -- see per_cam_operational below for
            # what it shows instead.
            per_cam_metrics = {}
            per_cam_operational = {}
            for cam_id in cam_ids:
                state = calib_states[cam_id]
                extra = []
                if app_state["mode"] == "calibrate":
                    if state.intrinsics_locked:
                        extra.append(f"intrinsics: loaded from cache "
                                     f"({state.reproj_error:.3f}px at save time)")
                    if state.last_detection_ts is not None:
                        silent_for = time.time() - state.last_detection_ts
                        if silent_for > runtime_cfg["no_detection_warning_s"] and detection_state["enabled"]:
                            extra.append(f"no detection for {silent_for:.0f}s")
                    elif state.connected:
                        extra.append("never detected the board yet")
                operational = []
                if cam_id in undistort_missing_intrinsics:
                    line = "undistort: enabled but no usable intrinsics -- showing raw frame"
                    extra.append(line)
                    operational.append(line)
                per_cam_operational[cam_id] = operational
                reproj = f"{state.reproj_error:.3f} px" if state.has_intrinsics_estimate else "n/a"
                per_cam_metrics[cam_id] = [
                    f"samples for int: {state.sample_count()}",
                    f"samples for ext: {pose_graph.sample_count_for(cam_id)}",
                    f"reproj error: {reproj}",
                    f"converged: {'yes' if state.converged else 'no'}",
                    *extra,
                ]

            # Live-preview panel's grid -- throttled well below the main
            # loop's own tick rate: JPEG-encoding+base64'ing every connected
            # camera's frame every tick would be wasted work/bandwidth for
            # something that only needs to look live, not match full fps.
            # Gated on show_grid since the grid is hidden (and Pass 1 above
            # isn't even decoding most cameras' previews) whenever it's
            # False.
            if show_grid and now - grid_state["last_update_mono"] >= 0.15:
                grid_state["last_update_mono"] = now
                grid_html.content = build_camera_grid_html(
                    cam_ids, latest_preview_frames, per_cam_metrics,
                    connected={c: sessions[c].connected for c in cam_ids},
                )

            if preview_toggle_state["on"] and capture_preview["selected"] is not None:
                capture_preview_status_md.content = "\n\n".join(
                    per_cam_operational.get(capture_preview["selected"], [])
                )

            last_pass2_ms = (time.perf_counter() - pass2_t0) * 1000

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
