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

This boundary is unchanged by calibration mode's per-camera
calibrate.DetectionWorker background threads (see main()'s calib_workers):
each camera's raw JPEG bytes are extracted from its dai queue message on
this main thread (AppCameraSession.poll(want_raw_bytes=True), a cheap
buffer copy, not a decode) -- only THEN, with no dai object involved at
all, do they cross into a worker thread for decode+ChArUco-detect+PnP-solve.
Workers never touch a dai.Device/pipeline/queue, and never touch a
CameraCalibState either (that stays exclusively main-thread-owned, read
AND written only here) -- every work item and result crossing the queue
boundary is a plain, self-contained snapshot (bytes/numpy arrays), so no
dai object or viser/GUI object is ever reachable from a second thread.
The periodic cv2.calibrateCamera pass gets the same treatment on its own
per-camera calibrate.RecalibrationWorker thread (main()'s recalib_workers)
-- it's expensive enough on its own (~200-400ms at realistic sample counts)
that running it inline, even throttled to one camera per tick, was
measured to reproduce the same main-thread-blocking symptom the detection
move was meant to fix. Same rule: no dai/viser object crosses the boundary,
only a plain snapshot (K/dist copies, a shallow list copy of accumulated
sample arrays), and CameraCalibState stays main-thread-owned.

PoseGraph.solve() (specifically _optimize_component's scipy.optimize.
least_squares) gets the same treatment on its own calibrate.PoseGraphWorker
thread (main()'s pose_graph_worker -- one, not per-camera, since there's
only one pose graph) -- measured live at ~120-160ms EVERY tick once the
graph's chain-consistency error crosses threshold, same symptom again.
Same rule: the worker only ever sees a snapshot of pose_graph.observations;
the real PoseGraph instance stays main-thread-owned.

One deliberate exception to "every dai call happens on main()'s thread":
AppCameraSession.start_calibration_pipeline registers a q_full.addCallback
(_on_full_arrival) -- this is depthai's OWN internal delivery mechanism,
not a thread this app spawns, and it fires the instant a message is
delivered rather than whenever this app's own tick-loop polling gets
around to it (measured on real hardware: ~15ms tighter on average, and
free of the systematic per-camera bias from this app's fixed cam0->cam6
polling order). Verified empirically before use, not assumed: an isolated
test against a real camera confirmed addCallback does NOT consume the
message (tryGet()/get() afterward still see it, so this can't race or
steal frames from the existing poll()-based flow) and always fired at or
before the same frame's tryGet()-visible arrival, never after. The
callback itself is deliberately minimal -- one time.time() read, one
thread-safe queue.Queue put, no dai calls, no decoding -- so it can never
block or back up depthai's own delivery pipeline; see
_pop_full_arrival_ts for how the main thread matches a callback-captured
timestamp back to a specific dequeued message.

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
import random
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
import flash_control
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


def draw_flash_not_seen_overlay(frame):
    """Whole-frame dark-orange tint, applied while the "Detect flash" toggle
    (flash_detect_state) is armed and this tick's ArUco check came back
    empty -- same tint-and-blend idiom and color convention as calibrate.
    draw_coverage_overlay's "undetected" cells (BGR (0, 90, 170)), just
    applied to the whole frame rather than a per-cell grid, since this is a
    single tag's current visible/not-visible state, not a spatial map.
    """
    overlay = np.full_like(frame, (0, 90, 170))
    return cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)


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
        self.last_frame_full_bytes = None  # raw JPEG bytes, see poll()'s want_raw_bytes
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
        self.preview_updated = False  # set each poll() call, see poll()'s decode_preview
        # (seq, host_ts_s) pairs fed by _on_full_arrival, a q_full callback
        # that fires on depthai's own internal thread the instant a message
        # is delivered -- see start_calibration_pipeline and
        # _pop_full_arrival_ts for why this exists (measured ~15ms tighter,
        # on average, than timestamping after this session's own tryGet()
        # notices the frame, and free of this app's fixed per-camera
        # polling-order bias). queue.Queue is thread-safe by design, so no
        # extra locking needed for this producer(callback thread)/
        # consumer(main thread) hand-off.
        self._full_arrival_queue = queue.Queue()

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
        self.q_full.addCallback(self._on_full_arrival)
        self.pipeline.start()

    def _on_full_arrival(self, _queue_name, msg):
        """q_full callback -- runs on depthai's OWN internal delivery thread
        (confirmed via isolated hardware test: not this session's main
        thread, and does NOT consume the message -- tryGet()/get() still see
        it afterward, so this is purely an additional notification, not a
        second consumer racing the real one). Deliberately minimal (one
        time.time() read, one thread-safe queue put, no dai calls, no
        decoding) so it can never back up depthai's own delivery pipeline
        for this or any other camera. See _pop_full_arrival_ts for how the
        main thread matches this back up to a specific dequeued message.
        """
        self._full_arrival_queue.put((msg.getSequenceNum(), time.time()))

    def _pop_full_arrival_ts(self, seq):
        """Host arrival time for full-res frame `seq`, from the callback
        above, instead of timestamping whenever this session's own poll()
        happens to notice the frame via tryGet() -- which, measured live,
        lags the true arrival by ~15ms on average (more under load) and
        carries a systematic per-camera bias from main()'s fixed
        cam0->cam6 polling order. Bounded search (a stale entry can
        accumulate here if q_full itself drops a frame before this session
        dequeues it) rather than an unbounded loop; falls back
        to time.time() if the callback hasn't caught up yet (should be rare
        to never, given the callback consistently fired first/simultaneously
        in testing) rather than block or misattribute a different frame's
        timestamp.
        """
        for _ in range(64):
            try:
                entry_seq, entry_ts = self._full_arrival_queue.get_nowait()
            except queue.Empty:
                break
            if entry_seq == seq:
                return entry_ts
            if entry_seq > seq:
                break  # shouldn't happen -- lost race, fall back below
        return time.time()

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
        resolution/fps/quality; column header), plus caller-supplied extra
        lines (take/session/calibration info) so this method itself doesn't
        need to know about that.

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
        for line in header_extra_lines:
            self.log_file.write(f"# {line}\n")
        self.log_file.write("# frame_log: host_ts_s sequence_num device_timestamp_s bytes\n")

        self.recording_active = True

    def end_recording(self):
        self.recording_active = False
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
        if self.writer is not None:
            self.writer.stop()
            self.writer = None

    def poll(self, decode_full=True, decode_preview=True, record=False, want_raw_bytes=False):
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

        want_raw_bytes=True (calibrate mode only -- see main()'s Pass 1)
        extracts the raw JPEG bytes (calibrate._packet_bytes, a cheap buffer
        copy, NOT a decode) into last_frame_full_bytes and returns without
        touching decode_full/cv2.imdecode at all -- the decode itself moves
        to a per-camera calibrate.DetectionWorker background thread instead,
        since it and everything downstream of it (ChArUco detection, PnP
        solve) has no dai.Device dependency (see module docstring). Ignored
        when record=True and already recording, same precedence decode_full
        has below.

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
        if preview_msg is not None:
            # Actual applied exposure (as opposed to what was last *requested*
            # via set_exposure) -- ImgFrame reports the real per-frame value,
            # confirmed live to update within a handful of frames of a
            # set_exposure() call. Used to give the exposure modal real
            # confirmation feedback instead of a fire-and-forget send. This is
            # a cheap metadata read on the already-dequeued message -- no
            # decode involved -- so it must run every tick a message arrives,
            # NOT only when decode_preview is True: decode_preview is a
            # display-only concern (Capture mode only decodes the ONE
            # selected camera's preview for showing on screen), and gating
            # this metadata read on it too meant every OTHER camera's
            # actual_shutter_us/actual_iso silently froze at whatever they
            # were the last time that camera's preview happened to be on
            # screen -- confirmed live: changing exposure visibly changed
            # every camera's brightness, but only the one selected camera's
            # reported values (and therefore the Camera Settings panel, and
            # each take's logged iso/shutter_us) ever updated to match.
            self.actual_shutter_us = round(preview_msg.getExposureTime().total_seconds() * 1_000_000)
            self.actual_iso = preview_msg.getSensitivity()
            if decode_preview:
                self.last_frame_preview = preview_msg.getCvFrame()
                self.preview_updated = True

        full_msg = self.q_full.tryGet()
        if full_msg is None:
            return False

        self.last_frame_full_ts = self._pop_full_arrival_ts(full_msg.getSequenceNum())
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

        if want_raw_bytes:
            self.last_frame_full_bytes = calibrate._packet_bytes(full_msg)
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


CAMERA_IDENTITY_PATH = os.path.join("logs", "camera_identity.json")


def load_camera_identity():
    """Tiny persisted cache: {"skip_oak_d": bool, "oak_d_device_id": str|None}.
    Backs show_resolution_picker's "Never even try OAK-D" checkbox -- that
    setting, and the OAK-D's device_id once learned (via one probe_camera_
    kind call), are remembered across app runs so a ticked box means the
    OAK-D is filtered out of device_infos before ANY connection attempt to
    it, on every future run, not just the one it was ticked in. Safe with
    no file yet (fresh install): both fields default to "unknown"/off.
    """
    if not os.path.exists(CAMERA_IDENTITY_PATH):
        return {"skip_oak_d": False, "oak_d_device_id": None}
    try:
        with open(CAMERA_IDENTITY_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"skip_oak_d": False, "oak_d_device_id": None}
    return {
        "skip_oak_d": bool(data.get("skip_oak_d", False)),
        "oak_d_device_id": data.get("oak_d_device_id"),
    }


def save_camera_identity(identity):
    os.makedirs(os.path.dirname(CAMERA_IDENTITY_PATH), exist_ok=True)
    with open(CAMERA_IDENTITY_PATH, "w", encoding="utf-8") as f:
        json.dump(identity, f, indent=2)


def show_resolution_picker(viser_mgr):
    """Shown BEFORE any config/camera/board setup exists -- the operator's
    choice here determines which calibrate_config*.yaml gets loaded, which
    in turn determines resolution, fps, and which camera pipeline variant
    (v2 native-sensor-binned vs v3 ISP-scale, see calibrate.
    build_calibration_pipeline) every camera uses for the rest of this run.

    Also shows a persistent "Never even try OAK-D" checkbox (see
    load_camera_identity/save_camera_identity) -- a debugging aid, added
    after a live "disable this camera" toggle in the exposure modal
    (closing a running camera's device on demand) was confirmed live this
    session to reliably crash the whole app: closing even one healthy
    device while others are concurrently open makes every other device
    miss its keepalive ping (device.close()'s own crash-dump-collection
    step appears to block long enough to cause this) and triggers a native
    depthai crash. Ticking this instead means the OAK-D is filtered out of
    device_infos before step_boot() ever connects it, on this run AND every
    future one (remembered on disk) -- there's then never a moment where it
    needs to be closed while others are live.

    This function itself NEVER touches a dai.Device -- no probing, no
    pinging, nothing -- so clicking a resolution button responds instantly
    regardless of how many cameras are plugged in or whether any are
    misbehaving. It only reads/writes the small camera_identity.json cache.
    The OAK-D's device_id is instead learned "for free" the first time it's
    actually connected during the real staggered boot (see main()'s
    step_boot -- it already opens a getConnectedCameraFeatures() check on a
    device it's connecting anyway, no extra connect/close needed), so
    ticking this box before the OAK-D's identity is known still boots it
    ONE more time, but never again after that.

    viser_mgr is already constructed at this point (see main()) using only
    a bootstrap port -- ViserManager.__init__ touches nothing else from cfg,
    so this reuses that same server/browser session rather than opening a
    second one once the real cfg is known. Blocks (plain queue.get(), no
    polling loop) since nothing else needs to progress while this is up --
    unlike show_calibration_picker, no camera is connected yet at this
    point for there to be boot progress to interleave.

    Returns (chosen_config_path, excluded_device_ids).
    """
    identity = load_camera_identity()
    choice_queue = queue.Queue()
    with viser_mgr.server.gui.add_modal("Session setup") as modal:
        viser_mgr.server.gui.add_markdown(
            "**Select a camera resolution/mode for this session.**\n\n"
            "This determines calibration too -- each profile has its own "
            "saved intrinsics, kept separate from the others."
        )
        for label, path in calibrate.RESOLUTION_PROFILES:
            btn = viser_mgr.server.gui.add_button(label)
            btn.on_click(lambda _, p=path: choice_queue.put(p))

        viser_mgr.server.gui.add_divider()
        skip_oak_d_cb = viser_mgr.server.gui.add_checkbox(
            "Never even try OAK-D (remembered across runs)",
            initial_value=identity["skip_oak_d"],
        )

    chosen_path = choice_queue.get()
    skip_oak_d = skip_oak_d_cb.value
    oak_d_device_id = identity["oak_d_device_id"]
    excluded_device_ids = {oak_d_device_id} if skip_oak_d and oak_d_device_id is not None else set()
    save_camera_identity({"skip_oak_d": skip_oak_d, "oak_d_device_id": oak_d_device_id})
    modal.close()
    return chosen_path, excluded_device_ids


# ==============================================================================
# Main
# ==============================================================================
def main():
    # Started as the very first thing in main() -- connecting over BLE
    # (scan + GATT connect) commonly takes several seconds, so starting it
    # here gives it the most possible time to settle in the background
    # before an operator ever reaches "Start capture", where low latency
    # actually matters (see _do_start_take's use of this). Doesn't depend
    # on cfg/dai/anything else, so there's no reason to wait.
    flash_conn = flash_control.FlashConnection()

    parser = argparse.ArgumentParser(description="Combined calibration + capture app (viser)")
    parser.add_argument(
        "--config", default=None,
        help="Path to the YAML config file (shared with calibrate.py). If omitted, a startup "
             "picker lets you choose a resolution profile (see calibrate.RESOLUTION_PROFILES).",
    )
    args = parser.parse_args()

    if dai is None:
        print("[Error] depthai is not installed in this environment.")
        return

    if args.config is not None:
        # Explicit --config skips the picker entirely -- unchanged bypass
        # behavior for scripted/automated runs. No camera-exclusion option
        # in this path either: every discovered device is used.
        cfg = calibrate.load_config(args.config)
        viser_mgr = calibrate.ViserManager(cfg)
        excluded_device_ids = set()
    else:
        # Bootstrap a viser server using only the default port -- ViserManager.
        # __init__ touches nothing else from cfg at construction time, so this
        # is safe before any resolution profile is known. Reused as-is (same
        # server/browser session, not a second one) once a profile is picked.
        viser_mgr = calibrate.ViserManager({"viser": {"port": calibrate.DEFAULT_CONFIG["viser"]["port"]}})
        chosen_path, excluded_device_ids = show_resolution_picker(viser_mgr)
        cfg = calibrate.load_config(chosen_path)
        viser_mgr.cfg = cfg

    print(f"[Viser] http://localhost:{viser_mgr.server.get_port()}")
    if psutil is not None:
        psutil.cpu_percent(interval=None)  # prime -- first call has no baseline, see update below
    _board, detector, board_points_3d = calibrate.build_board(cfg)
    _alignment_board, alignment_detector, alignment_board_points_3d = calibrate.build_board(cfg, "alignment_board")
    image_size = (cfg["camera"]["record_width"], cfg["camera"]["record_height"])

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
    if excluded_device_ids:
        device_infos = [info for info in device_infos if info.deviceId not in excluded_device_ids]
        print(f"[Camera] Excluded for this session: {', '.join(sorted(excluded_device_ids))}")
        if not device_infos:
            print("[Error] Every discovered camera was excluded for this session.")
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
    # Reloaded (not reused from show_resolution_picker) since that function
    # may not have run at all (--config bypass). Used below to learn the
    # OAK-D's device_id "for free" off an already-open connection, the
    # first time it's ever actually booted -- see step_boot.
    camera_identity = load_camera_identity()

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
            # Piggybacks on the connection just made above -- no extra
            # connect/close of its own (see show_resolution_picker's
            # docstring for why a standalone probe was removed: it made
            # the resolution picker block/look unresponsive). Only writes
            # camera_identity.json the first time this device_id is seen
            # as the OAK-D, so a normal boot's steady state does no extra
            # disk I/O.
            try:
                is_stereo = dai.CameraBoardSocket.CAM_B in {
                    f.socket for f in sess.device.getConnectedCameraFeatures()
                }
            except Exception:
                is_stereo = False
            if is_stereo and camera_identity.get("oak_d_device_id") != sess.device_id:
                camera_identity["oak_d_device_id"] = sess.device_id
                save_camera_identity(camera_identity)
                print(f"[Camera] Identified {cam_id} ({sess.device_id}) as the OAK-D -- "
                      f"cached for future 'Never even try OAK-D' runs.")
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
    # Bumped every time calib_states[cam_id] is reassigned (start_calibration,
    # start_new_calibration_live, "reset"/"uncache" commands) -- lets Pass 2
    # discard a DetectionWorker result computed against a since-discarded
    # CameraCalibState (in flight when the reset happened) instead of
    # applying a stale detection to the fresh one. See calib_workers below.
    calib_epoch = {cam_id: 0 for cam_id in cam_ids}
    # One background thread per camera, for the app's whole life -- has no
    # dai dependency and holds no CameraCalibState reference (see
    # calibrate.DetectionWorker), so it's independent of both step_boot()
    # and every calib_states reset above; started once, unconditionally.
    calib_workers = {cam_id: calibrate.DetectionWorker(cam_id, cfg) for cam_id in cam_ids}
    for worker in calib_workers.values():
        worker.start()
    # Same background-thread treatment for the periodic cv2.calibrateCamera
    # pass (measured at ~200-400ms for 30-60 samples, growing with sample
    # count) -- inline on the main thread this reproduces the exact
    # "drags fps down" symptom DetectionWorker was built to avoid, even
    # throttled to one camera per tick (tried live: didn't help, since total
    # calibrateCamera cost is unchanged by throttling, only its
    # distribution). recalib_in_flight tracks which cameras currently have
    # an outstanding job, so Pass 2 never submits a second one for the same
    # camera before the first's result is applied -- see RecalibrationWorker.
    recalib_workers = {cam_id: calibrate.RecalibrationWorker(cam_id) for cam_id in cam_ids}
    for worker in recalib_workers.values():
        worker.start()
    recalib_in_flight = set()
    pose_graph = calibrate.PoseGraph(cfg)
    # PoseGraph.solve() itself (specifically _optimize_component's
    # scipy.optimize.least_squares) gets the same background-thread
    # treatment -- measured live at ~120-160ms EVERY tick once the graph's
    # chain-consistency error crosses threshold, which reproduced the exact
    # "drags fps down" symptom the other two workers above were built to
    # avoid. One worker (not per-camera -- there's only one pose graph);
    # pose_solve_in_flight is the single-job-at-a-time gate, mirroring
    # recalib_in_flight; pose_graph_epoch is the single-pose-graph
    # equivalent of calib_epoch, bumped whenever pose_graph itself is
    # reset (clear_extrinsics_for), so a solve result computed against a
    # since-cleared graph gets discarded instead of resurrecting stale poses.
    pose_graph_worker = calibrate.PoseGraphWorker(cfg)
    pose_graph_worker.start()
    pose_graph_epoch = {"v": 0}
    pose_solve_in_flight = False
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
        for cam_id in cam_ids:
            calib_epoch[cam_id] += 1
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

    def _world_align_status_text():
        """Live status line shown under the "Set down direction from board"
        button -- without this, an armed request that can never resolve (most
        commonly: right after "Start new calibration" wipes pose_graph/
        pose_result, so no camera has a solved extrinsic pose yet) looks
        identical to the button silently doing nothing. See module docstring/
        request_world_alignment for why a solved pose is required at all.
        """
        if app_state["mode"] != "calibrate":
            return ("_Only available during a live calibration session -- "
                    "click 'Start new calibration' first._")
        if world_align["pending"]:
            if not detection_state["enabled"]:
                return "_Armed, but ChArUco detection is off -- turn detection back on to resolve it._"
            if pose_result is None or not pose_result["poses"]:
                return ("_Armed -- waiting for a solved camera pose. Show the MAIN "
                        "calibration board to 2+ cameras first, then the alignment board._")
            return "_Armed -- show the alignment board to any posed camera to set the world's origin/down direction._"
        if world_align["R"] is not None:
            return "_Down direction/origin already set._"
        return "_Not armed. Click the button, then show the alignment board._"

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

    # "Detect flash" toggle (calibration mode only, see the button near
    # calibration_detection_checkbox below). Detection itself runs off-
    # thread in calibrate.DetectionWorker (full resolution, deliberately
    # slow by user request rather than a faster downscaled search -- see
    # DetectionWorker.run()'s detect_flash branch), sharing the SAME per-
    # camera worker ChArUco detection already uses -- arming this forces
    # detection_state["enabled"] off (see _toggle_flash_detect) so the two
    # never compete for that one slow decode+detect slot. "seen" updates
    # at whatever cadence the worker manages, not persisted/latched beyond
    # that -- the workflow is aiming a camera in real time at a STATIONARY
    # LED, so the highlight must be able to flip back off once the tag
    # leaves frame, not stay stuck green from an earlier moment.
    flash_detect_state = {"armed": False, "seen": {cam_id: False for cam_id in cam_ids}}

    # Per-camera calibration enable/disable -- now driven entirely by the
    # sequential intrinsics-then-extrinsics workflow below (calib_phase_state/
    # the per-camera "Capture intrinsics" buttons and the "Capture
    # extrinsics" button -- see the "select_intrinsics_cam"/"start_extrinsics"
    # cmd_queue handlers). Starts all-False: no camera should
    # be actively detected until the operator explicitly picks one (see
    # calib_phase_state below) -- a bare False default here also protects
    # against ever accumulating pose-graph edges before any camera's
    # intrinsics have actually been captured.
    camera_enabled = {cam_id: False for cam_id in cam_ids}

    # Sequential calibration workflow: "intrinsics" phase has at most one
    # camera (active_cam) enabled at a time, picked via that camera's
    # "Capture intrinsics" button (which drives camera_enabled above);
    # "extrinsics" phase re-enables every camera at once (today's original
    # concurrent behavior) once every camera's intrinsics are done. See
    # the "select_intrinsics_cam"/"start_extrinsics" cmd_queue handlers.
    calib_phase_state = {"phase": "intrinsics", "active_cam": None}

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
        "phase": "idle",       # idle | countdown | starting | recording | stopping
        "countdown_start_mono": None,  # time.monotonic() when "countdown" began
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
        # Calibration mode's "Step 1" single-camera view -- mirrors
        # capture_preview_image/capture_preview_status_md above exactly,
        # but driven by calib_phase_state["active_cam"] instead of
        # capture_preview["selected"] (see the main loop's Pass 1/Pass 2).
        # A separate widget pair rather than reusing Capture mode's own,
        # since both modes can't show a camera at the same time anyway
        # (mutually exclusive via ui_mode_state) but keeping them distinct
        # avoids any cross-mode visibility-toggling subtlety.
        intrinsics_preview_image = viser_mgr.server.gui.add_image(
            np.zeros((4, 4, 3), dtype=np.uint8), label="live", visible=False,
        )
        intrinsics_preview_status_md = viser_mgr.server.gui.add_markdown("", visible=False)
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

            # Rig-setup aid: the sync LED is mounted in a stationary spot,
            # and an operator aims/adjusts each camera (or the LED's
            # placement) until every camera can see its ArUco tag. A plain
            # on/off toggle, not "start/reset" -- see flash_detect_state's
            # comment for why this needs to be live/instantaneous rather
            # than a one-way latch.
            detect_flash_btn = viser_mgr.server.gui.add_button("Detect flash")

            def _toggle_flash_detect(_):
                flash_detect_state["armed"] = not flash_detect_state["armed"]
                if flash_detect_state["armed"]:
                    # Flash and ChArUco detection share one per-camera
                    # DetectionWorker/raw-bytes pipeline (see need_raw_bytes
                    # in Pass 1) and the worker only ever does one or the
                    # other per submission (detect_flash) -- forcing ChArUco
                    # off here, not just leaving it independent, avoids
                    # quietly wasting the worker's slow full-res decode+
                    # detect budget on a board search the operator isn't
                    # even looking at while aiming for the flash tag.
                    # _set_detection is defined later in main() but already
                    # exists in this closure's scope by the time a button
                    # click can actually fire it.
                    flash_detect_state["pre_armed_detection_enabled"] = detection_state["enabled"]
                    _set_detection(False)
                else:
                    _set_detection(flash_detect_state.get("pre_armed_detection_enabled", True))
                detect_flash_btn.label = (
                    "Stop detecting flash" if flash_detect_state["armed"] else "Detect flash"
                )
                detect_flash_btn.color = "red" if flash_detect_state["armed"] else "gray"

            detect_flash_btn.on_click(_toggle_flash_detect)
            flash_detect_status_md = viser_mgr.server.gui.add_markdown("")

            viser_mgr.add_world_alignment_button(request_world_alignment)
            align_status_md = viser_mgr.server.gui.add_markdown(_world_align_status_text())

            save_button = viser_mgr.server.gui.add_button("Save now", color="green")
            save_button.on_click(lambda _: cmd_queue.put("save"))

            viser_mgr.server.gui.add_divider()

            # Step 1: intrinsics, one camera at a time -- clicking a camera's
            # button turns OFF board detection on every other camera (via
            # camera_enabled, see calib_phase_state above) and brings that
            # camera up as the only live view (see intrinsics_preview_image
            # below and the show_grid/decode_this_preview changes in the
            # main loop). Label/color are refreshed every tick in the main
            # loop (booting/gray "Capture intrinsics" / red "Capturing..." /
            # green "done"), not just on click, since convergence happens
            # asynchronously on a background RecalibrationWorker.
            viser_mgr.server.gui.add_markdown("**Step 1: Capture intrinsics (one camera at a time)**")
            intrinsics_btns = {}
            for cam_id in cam_ids:
                btn = viser_mgr.server.gui.add_button(f"Capture intrinsics: {cam_id}")

                def _toggle_intrinsics_cam(_, c=cam_id):
                    if calib_phase_state["active_cam"] == c:
                        cmd_queue.put("select_intrinsics_cam none")
                    else:
                        cmd_queue.put(f"select_intrinsics_cam {c}")

                btn.on_click(_toggle_intrinsics_cam)
                intrinsics_btns[cam_id] = btn

            # Step 2: extrinsics, all cameras live together -- only
            # clickable once every camera shows converged/cached intrinsics
            # (see the all_intrinsics_done check in the main loop). Reusable
            # even after extrinsics capture has started (re-click wipes the
            # pose graph and starts over), and a camera's own intrinsics
            # button above stays live throughout -- clicking it drops back
            # into intrinsics capture for just that camera and discards its
            # pose-graph edges (see "select_intrinsics_cam"'s handler).
            viser_mgr.server.gui.add_markdown("**Step 2: Capture extrinsics (all cameras)**")
            start_extrinsics_btn = viser_mgr.server.gui.add_button("Capture extrinsics", color="green")
            start_extrinsics_btn.on_click(lambda _: cmd_queue.put("start_extrinsics"))
            start_extrinsics_status_md = viser_mgr.server.gui.add_markdown("")
            extrinsics_guidance_md = viser_mgr.server.gui.add_markdown("")

            viser_mgr.server.gui.add_divider()

            accept_anyway_btns = {}
            for cam_id in cam_ids:
                reset_btn = viser_mgr.server.gui.add_button(f"Reset {cam_id}")
                reset_btn.on_click(lambda _, c=cam_id: cmd_queue.put(f"reset {c}"))
                # Manual override for a camera whose reprojection error
                # plateaus above reproj_error_threshold_px no matter how much
                # more data it gets (a data-quality floor -- board rigidity,
                # residual motion blur, lens distortion model mismatch --
                # not something more samples fixes). Caches its BEST current
                # K/dist as-is, bypassing the quality gate for just this one
                # camera rather than lowering reproj_error_threshold_px for
                # everyone. Disabled/relabeled every tick (see the main
                # loop) once there's nothing to accept yet or it's already
                # done.
                accept_anyway_btn = viser_mgr.server.gui.add_button(f"Accept {cam_id} intrinsics anyway")
                accept_anyway_btn.on_click(lambda _, c=cam_id: cmd_queue.put(f"force_accept_intrinsics {c}"))
                accept_anyway_btns[cam_id] = accept_anyway_btn

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
            # TOTAL duration of the 3/2/1 countdown shown before a take
            # actually starts (see capture_state["phase"] == "countdown" in
            # the main loop) -- e.g. the default 1s means "3", "2", "1" each
            # show for 1/3s, the whole countdown taking 1s. Read directly off
            # the slider every tick in the main loop (a plain value read, not
            # a dai call, so no queueing needed -- same as how the exposure
            # modal's sliders are read straight from their on_click handler).
            capture_delay_slider = viser_mgr.server.gui.add_slider(
                "Countdown duration (s)", min=0.0, max=5.0, step=0.5, initial_value=1.0,
            )
            viser_mgr.server.gui.add_divider()
            flash_banner = viser_mgr.server.gui.add_html("", visible=False)
            flash_banner_state = {"html": ""}
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

        grid_html.visible = mode == "Calibration" and calib_phase_state["active_cam"] is None
        preview_toggle_checkbox.visible = mode == "Capture"
        capture_preview_image.visible = mode == "Capture"
        capture_preview_status_md.visible = mode == "Capture"
        intrinsics_preview_image.visible = mode == "Calibration" and calib_phase_state["active_cam"] is not None
        intrinsics_preview_status_md.visible = mode == "Calibration" and calib_phase_state["active_cam"] is not None
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
            pose_graph_epoch["v"] += 1
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
            if stale:
                pose_graph_epoch["v"] += 1
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
            calib_epoch[cam_id] += 1
            viser_mgr.hide_board_pose(cam_id)
            camera_enabled[cam_id] = False
        calib_phase_state["phase"] = "intrinsics"
        calib_phase_state["active_cam"] = None
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
            # What timing-related raw data this take has, independent of how
            # it's later aligned (align_session.py's `timebase` records the
            # latter) -- see alignment_improvements.md.
            "depthai_version": dai.__version__ if dai is not None else None,
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

        Fires the sync LED's "S" sequence right at the top, over the
        persistent BLE connection opened at app startup (flash_conn, see
        main()) -- send_command_nowait is a single GATT write on an
        already-open connection and returns immediately, not a fresh
        multi-second scan+connect, so this stays safe to call straight from
        this otherwise-cheap, main-thread function without a wrapper
        thread. Called with a small delay_s so the actual GATT write (and
        the firmware's own 1s wait before the white flash on top of that)
        happens comfortably after the per-camera begin_recording loop below
        has actually started every camera recording, rather than racing it
        -- the delay itself runs on flash_conn's own BLE thread via
        asyncio.sleep, so it costs nothing on this main thread. Logs its
        own success/failure (including "not connected yet") -- never
        raises, never delays the take if the device isn't reachable.

        No per-camera clock-offset calibration happens here anymore
        (removed -- see git history): the default alignment path
        (align_session.py) already trusts each camera's own depthai-
        synced device_ts_s directly, and that per-camera BLOCKING frame
        read, done sequentially for every camera right at take-start, was
        exactly the kind of main-thread stall that can make OTHER devices
        miss their depthai keepalive ping and get disconnected --
        confirmed live this session as the cause of a native crash when
        starting an 8-camera take.
        """
        flash_conn.send_command_nowait("S", delay_s=0.2)

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
                                calib_epoch[cam_id] += 1
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
                                calib_epoch[cam_id] += 1
                                print(f"[Reset] {cam_id} calibration state cleared (intrinsics uncached).")
                        if evicted and cfg["intrinsics_cache"]["enabled"]:
                            calibrate._atomic_write_json(cfg["intrinsics_cache"]["path"], intrinsics_cache)
                        clear_extrinsics_for(target)
                    elif cmd.startswith("force_accept_intrinsics"):
                        # "Accept anyway" (see calibration_folder above) --
                        # deliberate quality-gate bypass for a camera whose
                        # reproj_error plateaus above the threshold. Caches
                        # whatever K/dist/reproj_error it currently has as-is;
                        # build_cache_entry/upsert_intrinsics_cache do no
                        # thresholding of their own, so this is the same cache
                        # write a real convergence would do, just triggered
                        # manually instead of by apply_recalibration_result.
                        #
                        # Sets intrinsics_locked (not just converged) --
                        # confirmed live as a real bug otherwise: a
                        # RecalibrationWorker job already in flight (or still
                        # eligible to fire again, since due_for_recalibrate
                        # only checks intrinsics_locked, never converged) can
                        # land its result AFTER this click and have
                        # apply_recalibration_result silently recompute
                        # `converged` from the real thresholds, reverting this
                        # override back to False with no further action from
                        # the operator. intrinsics_locked short-circuits
                        # accept_sample/due_for_recalibrate the same way a
                        # cache load does, fully freezing this camera; bumping
                        # calib_epoch additionally discards any result from a
                        # job already in flight before this click (the same
                        # protection "reset"/"uncache" already rely on).
                        target = cmd.split()[1]
                        state = calib_states.get(target)
                        if state is None or state.reproj_error == float("inf"):
                            print(f"[Calibration] {target}: no calibration result yet -- nothing to accept.")
                        elif state.intrinsics_locked:
                            print(f"[Calibration] {target}: already cached -- nothing to do.")
                        else:
                            state.converged = True
                            state.intrinsics_locked = True
                            state.manually_accepted = True
                            calib_epoch[target] += 1
                            calibrate.upsert_intrinsics_cache(cfg, sessions[target].device_id, state, intrinsics_cache)
                            print(f"[Calibration] {target}: intrinsics manually accepted at "
                                  f"{state.reproj_error:.3f}px (quality gate is "
                                  f"{cfg['intrinsics']['reproj_error_threshold_px']:.2f}px) -- locked.")
                    elif cmd.startswith("select_intrinsics_cam"):
                        # Step 1's per-camera buttons (see calibration_folder
                        # above). "none" deactivates back to the idle grid
                        # view with every camera disabled; a cam_id makes it
                        # the only enabled (and only visible) camera. Clicking
                        # a cam_id while already in the extrinsics phase drops
                        # back to the intrinsics phase for just that camera
                        # and discards its pose-graph edges (they're built on
                        # intrinsics that are about to change) -- the operator
                        # returns to extrinsics by clicking "Capture
                        # extrinsics" again, which fully re-wipes the pose
                        # graph anyway (see "start_extrinsics" below), so no
                        # stale edge from this camera can survive into that.
                        parts = cmd.split()
                        target = parts[1] if len(parts) > 1 else "none"
                        was_extrinsics_phase = calib_phase_state["phase"] == "extrinsics"
                        calib_phase_state["phase"] = "intrinsics"
                        if target == "none":
                            calib_phase_state["active_cam"] = None
                            for c in cam_ids:
                                camera_enabled[c] = False
                                calib_states[c].last_edge_pose = None
                                viser_mgr.hide_board_pose(c)
                            print("[Calibration] Intrinsics capture stopped -- no camera active.")
                        else:
                            calib_phase_state["active_cam"] = target
                            for c in cam_ids:
                                camera_enabled[c] = (c == target)
                                if c != target:
                                    calib_states[c].last_edge_pose = None
                                    viser_mgr.hide_board_pose(c)
                            if was_extrinsics_phase:
                                clear_extrinsics_for(target)
                            print(f"[Calibration] Capturing intrinsics for {target} only.")
                    elif cmd == "start_extrinsics":
                        # Step 2's button (see calibration_folder above) --
                        # re-enables every camera at once (today's original
                        # concurrent behavior) and fully wipes the pose graph
                        # (clear_extrinsics_for("all"), also used by "Start
                        # new calibration") so extrinsics starts from a clean
                        # slate built only on each camera's now-finalized
                        # intrinsics, never on stale pre-convergence edges.
                        # Re-clickable even while already in this phase, to
                        # deliberately restart extrinsics capture from scratch.
                        calib_phase_state["phase"] = "extrinsics"
                        calib_phase_state["active_cam"] = None
                        for c in cam_ids:
                            camera_enabled[c] = True
                        clear_extrinsics_for("all")
                        print("[Calibration] Extrinsics capture started -- all cameras live.")
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
                    if capture_delay_slider.value <= 0:
                        # 0 skips the countdown phase entirely rather than
                        # showing a zero-length "3...2...1..." flash.
                        capture_state["phase"] = "starting"
                        capture_btn.label, capture_btn.disabled, capture_btn.color = "Starting...", True, "red"
                        capture_status_md.content = "_Starting take..._"
                    else:
                        capture_state["phase"] = "countdown"
                        capture_state["countdown_start_mono"] = time.monotonic()
                        capture_btn.label, capture_btn.disabled, capture_btn.color = "Starting...", True, "red"
                elif cmd == "stop" and capture_state["phase"] == "recording":
                    capture_state["phase"] = "stopping"
                    capture_btn.label, capture_btn.disabled, capture_btn.color = "Stopping...", True, "green"
                    capture_status_md.content = "_Stopping..._"

            if capture_state["phase"] == "countdown":
                # 3/2/1 countdown before a take actually starts, giving
                # whoever's in frame time to get ready -- see
                # capture_delay_slider above (the "Countdown interval (s)"
                # setting). Read live every tick rather than snapshotted at
                # "start" click, same as every other slider-driven setting
                # in this file. Transitions into the existing "starting"
                # phase below once the full 3-step countdown elapses, which
                # then proceeds exactly as it always has.
                total = capture_delay_slider.value
                step = total / 3.0
                elapsed = time.monotonic() - capture_state["countdown_start_mono"]
                if elapsed >= total:
                    capture_state["phase"] = "starting"
                else:
                    count = 3 - min(2, int(elapsed // step))
                    capture_btn.label = f"Starting in {count}..."
                    capture_status_md.content = f"# {count}"

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
            flash_problems = []
            for take in capture_state["takes"]:
                if take["stop_time_unix"] is not None:
                    length_str = f"{take['stop_time_unix'] - take['start_time_unix']:.0f}s"
                else:
                    length_str = f"{time.time() - take['start_time_unix']:.0f}s (recording)"
                status = postprocess.read_status(take["take_dir"])
                status_str = postprocess.status_text(status)
                row = take_row_widgets[take["take_n"]]
                row["label_md"].content = f"Take {take['take_n']}: {length_str} -- {status_str}"
                row["play_btn"].visible = status["status"] == "done"
                text, problem = postprocess.flash_sync_text(status)
                if problem:
                    flash_problems.append(f"Take {take['take_n']}: {text}")
            # Red banner (shown only while some take of this session has a flash-sync problem) so a missing
            # flash can't be overlooked among the per-take lines -- the operator can re-record right away.
            banner_html = ""
            if flash_problems:
                banner_html = (
                    '<div style="background:#b00020;color:#fff;padding:8px 10px;border-radius:4px;'
                    'font-size:13px;line-height:1.4"><b>&#9888; FLASH SYNC PROBLEM</b><br>'
                    + "<br>".join(html.escape(line) for line in flash_problems) + "</div>"
                )
            if banner_html != flash_banner_state["html"]:
                flash_banner_state["html"] = banner_html
                flash_banner.content = banner_html
                flash_banner.visible = bool(banner_html)

            # Same status refresh for the Viewer folder's past-session takes
            # -- length was already computed once at startup (see
            # viewer_folder above), only postprocess status can still
            # change live.
            for take_dir, take_n, length_str, label_md in viewer_take_rows:
                status_str = postprocess.status_text(postprocess.read_status(take_dir))
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

            # Which live-preview view is showing -- driven by
            # ui_mode_state["current"] (see _set_ui_mode, which owns the
            # initial/on-mode-change .visible toggling for every live-view
            # widget) AND, within Calibration mode, by calib_phase_state --
            # while a camera is actively being intrinsics-captured, it gets
            # the ONLY live view (see intrinsics_active below), not the grid,
            # mirroring Capture mode's own single-camera behavior. Unlike
            # ui_mode_state (only changes on a mode-button click), active_cam
            # can change every tick (a "Capture intrinsics" button click), so
            # these three widgets' .visible is refreshed here every tick
            # rather than only in _set_ui_mode.
            intrinsics_active = (
                ui_mode_state["current"] == "Calibration" and calib_phase_state["active_cam"] is not None
            )
            show_grid = ui_mode_state["current"] == "Calibration" and not intrinsics_active
            show_single = ui_mode_state["current"] == "Capture"
            if ui_mode_state["current"] == "Calibration":
                grid_html.visible = show_grid
                intrinsics_preview_image.visible = intrinsics_active
                intrinsics_preview_status_md.visible = intrinsics_active

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
                # "loaded" mode's inline sanity-check detection still decodes
                # here (app.py:2197ish, out of scope for the worker-thread
                # move -- it's cheap relative to calibrate mode's
                # process_detection/maybe_recalibrate accumulation cost, see
                # module docstring); "calibrate" mode instead hands off raw
                # bytes to that camera's calibrate.DetectionWorker below, so
                # decode+detect happens off this thread.
                need_full_decode = (
                    not recording_now
                    and detection_state["enabled"]
                    and app_state["mode"] == "loaded"
                )
                need_raw_bytes = (
                    not recording_now
                    and app_state["mode"] == "calibrate"
                    # Flash-marker detection also needs raw bytes -- it's handed to
                    # the SAME per-camera DetectionWorker as ChArUco detection, just
                    # with detect_flash=True so the worker runs a plain ArUco search
                    # instead (full resolution, deliberately slow -- see submit()'s
                    # call site and DetectionWorker.run()'s detect_flash branch for
                    # why). The two are mutually exclusive in practice (arming
                    # "Detect flash" forces detection_state["enabled"] off, see
                    # _toggle_flash_detect), but written as an OR so either one
                    # alone is sufficient to trigger a submission.
                    and (detection_state["enabled"] or flash_detect_state["armed"])
                )
                decode_this_preview = previews_on and (
                    show_grid or (show_single and cam_id == capture_preview["selected"])
                    or flash_detect_state["armed"]
                    or cam_id == calib_phase_state["active_cam"]
                )
                try:
                    got_frame = session.poll(
                        decode_full=need_full_decode, decode_preview=decode_this_preview,
                        record=recording_now, want_raw_bytes=need_raw_bytes,
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
                    if flash_detect_state["armed"] and not flash_detect_state["seen"].get(cam_id, False):
                        # Detection itself runs on the full-res frame below (not this
                        # preview frame -- see need_full_decode's comment on why); this
                        # just reads that result to decide whether to tint for display.
                        # One-tick-stale on the very first armed tick (negligible --
                        # a single ~30-60ms lag, not a correctness issue) since the
                        # full-res block below hasn't updated "seen" yet this iteration.
                        preview_frame = draw_flash_not_seen_overlay(preview_frame)
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
                    if cam_id == calib_phase_state["active_cam"]:
                        # Same resize-to-panel-width treatment as
                        # capture_preview_image above, for the same reason.
                        # Horizontally mirrored (display only -- the coverage
                        # overlay above was already baked into preview_frame
                        # before this flip, so it mirrors along with the
                        # image and stays aligned; the raw full-res bytes
                        # DetectionWorker/accept_sample actually use for
                        # intrinsics are untouched, read separately from
                        # session.last_frame_full_bytes) so moving the board
                        # left/right on-screen matches the operator's own
                        # left/right while they're looking at the display
                        # instead of the camera.
                        intr_frame = cv2.resize(
                            preview_frame, (960, 540), interpolation=cv2.INTER_LINEAR,
                        )
                        intr_frame = cv2.flip(intr_frame, 1)
                        intrinsics_preview_image.image = cv2.cvtColor(intr_frame, cv2.COLOR_BGR2RGB)

                if need_full_decode and got_frame and session.last_frame_full is not None:
                    fresh_frames[cam_id] = (session.last_frame_full, session.last_frame_full_ts)

                if (need_raw_bytes and got_frame and session.last_frame_full_bytes is not None
                        and camera_enabled.get(cam_id, True)):
                    calib_workers[cam_id].submit(
                        epoch=calib_epoch[cam_id],
                        frame_ts=session.last_frame_full_ts,
                        jpeg_bytes=session.last_frame_full_bytes,
                        K=calib_states[cam_id].K.copy(),
                        dist=calib_states[cam_id].dist.copy(),
                        do_alignment=world_align["pending"],
                        detect_flash=flash_detect_state["armed"],
                    )

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
                # Decode+detect+PnP-solve for this mode runs on a per-camera
                # calibrate.DetectionWorker background thread (submitted in
                # Pass 1 above) -- this just drains whatever result is
                # waiting and applies it, exactly like the old inline
                # decode+detect body used to, just fed from a queue instead
                # of computing the detection itself. A result whose epoch
                # doesn't match calib_epoch[cam_id] was computed against a
                # since-reset CameraCalibState (e.g. "Start new calibration"
                # fired while it was in flight) and is discarded.
                drained_count = 0
                # Cameras whose last_edge_pose is genuinely fresh THIS tick
                # -- see the pairwise correlation loop below, which must not
                # re-add an edge observation built from a pair of
                # last_edge_pose values neither of which actually changed
                # since the last tick.
                fresh_edge_cam_ids = set()
                for cam_id in cam_ids:
                    if not camera_enabled.get(cam_id, True):
                        viser_mgr.hide_board_pose(cam_id)
                        continue
                    result = calib_workers[cam_id].try_get_result()
                    if result is None or result["epoch"] != calib_epoch[cam_id]:
                        continue
                    drained_count += 1
                    state = calib_states[cam_id]
                    frame_ts = result["frame_ts"]

                    if result["did_alignment"]:
                        if result["align_pose"] is not None:
                            a_R, a_t, _a_err = result["align_pose"]
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

                    if result["ids"] is None:
                        viser_mgr.hide_board_pose(cam_id)
                        continue
                    corners2d, ids = result["corners2d"], result["ids"]

                    if result["pose"] is not None:
                        R, t, err = result["pose"]
                        calibrate.accept_sample(state, corners2d, ids, board_points_3d, R, t, cfg)
                    if result["pose"] is None:
                        viser_mgr.hide_board_pose(cam_id)
                        continue
                    R, t, err = result["pose"]
                    state.last_detection_ts = frame_ts
                    state.last_edge_pose = (frame_ts, R, t, err)
                    fresh_edge_cam_ids.add(cam_id)
                    state.last_warned_no_detection = False

                    if pose_result is not None and cam_id in pose_result["poses"]:
                        T_cam_to_world = calibrate.invert_T(pose_result["poses"][cam_id])
                        viser_mgr.update_board_pose(cam_id, T_cam_to_world @ calibrate.rt_to_T(R, t))
                    else:
                        viser_mgr.hide_board_pose(cam_id)

                # cv2.calibrateCamera itself runs on a per-camera
                # RecalibrationWorker background thread -- submitting a job
                # is cheap (a list copy, no cv2 call), so every due camera
                # can be submitted the same tick; recalib_in_flight is the
                # only thing preventing a camera being submitted twice
                # before its previous result is applied.
                for cam_id in cam_ids:
                    state = calib_states[cam_id]
                    if cam_id not in recalib_in_flight and calibrate.due_for_recalibrate(state, cfg):
                        recalib_in_flight.add(cam_id)
                        recalib_workers[cam_id].submit(
                            epoch=calib_epoch[cam_id],
                            object_points=list(state.object_points),
                            image_points=list(state.image_points),
                            image_size=state.image_size,
                            K=state.K, dist=state.dist,
                            has_intrinsics_estimate=state.has_intrinsics_estimate,
                        )

                for cam_id in cam_ids:
                    result_msg = recalib_workers[cam_id].try_get_result()
                    if result_msg is None:
                        continue
                    recalib_in_flight.discard(cam_id)
                    if result_msg["epoch"] != calib_epoch[cam_id]:
                        continue
                    state = calib_states[cam_id]
                    was_converged = state.converged
                    calibrate.apply_recalibration_result(state, result_msg["result"], cfg)
                    if cfg["intrinsics_cache"]["enabled"] and state.converged and not was_converged:
                        calibrate.upsert_intrinsics_cache(cfg, sessions[cam_id].device_id, state, intrinsics_cache)

                last_pass2_cams = drained_count

                # Each camera's last_edge_pose PERSISTS across ticks (only
                # overwritten when that camera's own DetectionWorker
                # produces a fresh result -- see fresh_edge_cam_ids above),
                # but this loop itself runs every tick. With independent
                # per-camera async detection now decoupled from the tick
                # rate, most ticks see NEITHER side of most pairs change --
                # without the fresh_edge_cam_ids guard below, the exact
                # same (cam_a, cam_b) observation (same timestamps, same
                # R/t) would get added to pose_graph.observations again on
                # every one of those ticks until one side finally refreshes,
                # flooding the graph with duplicates of whatever detection
                # happens to sit stale the longest and skewing
                # aggregate_edges' mean toward it. Requiring at least one
                # side to be fresh THIS tick restores the old synchronous
                # code's implicit guarantee (every tick's data was
                # genuinely new, since all cameras refreshed together) --
                # reusing cam_a's still-timestamp-valid pose against a
                # DIFFERENT, newly-fresh cam_b is still a legitimate distinct
                # observation and stays allowed, only a bit-for-bit repeat
                # of an already-recorded pair is excluded.
                for i, cam_a in enumerate(cam_ids):
                    edge_a = calib_states[cam_a].last_edge_pose
                    if edge_a is None:
                        continue
                    ts_a, R_a, t_a, err_a = edge_a
                    for cam_b in cam_ids[i + 1:]:
                        if cam_a not in fresh_edge_cam_ids and cam_b not in fresh_edge_cam_ids:
                            continue
                        edge_b = calib_states[cam_b].last_edge_pose
                        if edge_b is None:
                            continue
                        ts_b, R_b, t_b, err_b = edge_b
                        if abs(ts_a - ts_b) <= tol_s:
                            pose_graph.add_observation(cam_a, R_a, t_a, err_a, cam_b, R_b, t_b, err_b)

                if len(cam_ids) >= 2:
                    # pose_graph.solve() itself runs on a background
                    # PoseGraphWorker thread -- measured live at 120-160ms
                    # EVERY tick once the graph's chain-consistency error
                    # crosses naive_chain_error_threshold_deg (triggering
                    # _optimize_component's scipy.optimize.least_squares),
                    # which reproduced the exact "drags fps down" symptom
                    # DetectionWorker/RecalibrationWorker were built to
                    # avoid. Same split: main thread hands over a snapshot
                    # of pose_graph.observations, applies the result once
                    # ready. pose_solve_in_flight prevents submitting a
                    # second job before the first's result is applied --
                    # pose_result is left at its previous value (or None,
                    # before the first solve ever completes) on ticks with
                    # no fresh result, same "freeze, don't disappear"
                    # precedent used elsewhere in this loop.
                    if not pose_solve_in_flight:
                        pose_solve_in_flight = True
                        pose_graph_worker.submit(
                            epoch=pose_graph_epoch["v"],
                            observations_snapshot={k: list(v) for k, v in pose_graph.observations.items()},
                            cam_ids=cam_ids,
                        )
                    # apply_world_alignment mutates pose_result["poses"] IN
                    # PLACE and must only ever run once per FRESH, unaligned
                    # solve (see its own docstring) -- re-running it on a
                    # pose_result that's already been aligned (e.g. on a
                    # tick where the worker hasn't produced a new result
                    # yet, so pose_result is the same object as last tick)
                    # would compose the realignment transform on top of
                    # itself, visibly flip-flopping the displayed world
                    # between single- and double-aligned every other tick.
                    # So the standing-alignment re-apply below only runs
                    # inside "a genuinely new result just arrived" -- never
                    # unconditionally every tick like the old synchronous
                    # code could get away with (it built a brand-new
                    # pose_result every tick by construction).
                    pg_result_msg = pose_graph_worker.try_get_result()
                    if pg_result_msg is not None:
                        pose_solve_in_flight = False
                        if pg_result_msg["epoch"] == pose_graph_epoch["v"]:
                            pose_result = pg_result_msg["result"]
                            calibrate.apply_world_alignment(pose_result, world_align["R"], world_align["t"])

                    # Unlike the re-apply above, this only READS pose_result
                    # (world_align["R"]/["t"] start out None, so
                    # apply_world_alignment no-ops until the block below
                    # sets them) -- so it's safe, and worth doing, to check
                    # every tick against whatever pose_result currently is
                    # (this tick's fresh one, or the last fresh one), rather
                    # than only on ticks with a brand-new solve. The
                    # alignment-board detection itself is fleeting (2.0s
                    # freshness window, calib_states[cam_id].
                    # last_alignment_pose refreshed independently by
                    # DetectionWorker) -- gating this check to "only as
                    # often as a new pose-graph solve lands" meant missing
                    # that window more often than necessary, measured live
                    # to add several extra seconds to how long "Set down
                    # direction from board" took to resolve.
                    if world_align["pending"] and pose_result is not None:
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
                            # world_align["R"] was None on every prior tick
                            # (that's what gated this block from running at
                            # all until now), so pose_result -- however
                            # stale its OBJECT reference is -- has never
                            # been touched by apply_world_alignment before;
                            # applying it here, once, right as R/t are set,
                            # gives immediate visual feedback instead of
                            # waiting for the next fresh solve.
                            calibrate.apply_world_alignment(pose_result, world_align["R"], world_align["t"])
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

                    for cam_id in cam_ids:
                        state = calib_states[cam_id]
                        if pose_result is not None and cam_id in pose_result["poses"]:
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

            # Deliberately OUTSIDE the detection_state["enabled"]-gated block
            # above -- arming "Detect flash" forces that flag off (see
            # _toggle_flash_detect), so draining flash results needs its own,
            # independent path rather than living inside a block that's
            # skipped exactly when this matters. Just a non-blocking mailbox
            # read per camera (calibrate.DetectionWorker.try_get_result) --
            # the submission side (Pass 1 above) already decides whether each
            # worker is doing ChArUco or flash detection via detect_flash.
            if flash_detect_state["armed"]:
                for cam_id in cam_ids:
                    result = calib_workers[cam_id].try_get_result()
                    if result is not None and result.get("flash_seen") is not None:
                        flash_detect_state["seen"][cam_id] = result["flash_seen"]

            align_status_md.content = _world_align_status_text()

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
                if picked is None and capture_preview["selected"] is None:
                    # No frustum positions to pick a "middle" camera from --
                    # happens when uncalibrated (frustums come from
                    # extrinsics), confirmed live: this left Capture mode's
                    # single-camera preview stuck on "waiting for data"
                    # forever with no way to ever select a camera. Fall back
                    # to a random CONNECTED camera instead. Guarded on
                    # `selected is None` so this only rolls once, not every
                    # tick -- the disconnect check above already resets back
                    # to "auto" (re-triggering this fallback) if that camera
                    # later drops.
                    connected_cams = [c for c in cam_ids if sessions[c].connected]
                    if connected_cams:
                        picked = random.choice(connected_cams)
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
            # deliberately NOT repeated here as a percentage: the blue/orange
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
                ic = cfg["intrinsics"]
                extra = []
                if app_state["mode"] == "calibrate":
                    if state.manually_accepted:
                        # force_accept_intrinsics also sets intrinsics_locked
                        # (see its cmd_queue handler for why), so this check
                        # must come before the plain intrinsics_locked one
                        # below -- otherwise a manual accept would silently
                        # look identical to a real cache load.
                        extra.append(
                            f"intrinsics: manually accepted at {state.reproj_error:.3f}px "
                            f"(above the {ic['reproj_error_threshold_px']:.2f}px bar)"
                        )
                    elif state.intrinsics_locked:
                        extra.append(f"intrinsics: loaded from cache "
                                     f"({state.reproj_error:.3f}px at save time)")
                    if state.last_detection_ts is not None:
                        silent_for = time.time() - state.last_detection_ts
                        if silent_for > runtime_cfg["no_detection_warning_s"] and detection_state["enabled"]:
                            extra.append(f"no detection for {silent_for:.0f}s")
                    elif state.connected:
                        extra.append("never detected the board yet")
                    if calib_phase_state["phase"] == "intrinsics":
                        if cam_id == calib_phase_state["active_cam"]:
                            extra.append("intrinsics: capturing now (active)")
                        elif state.intrinsics_locked or state.converged:
                            extra.append("intrinsics: done")  # detail (cache/manual) already added above
                        else:
                            extra.append("intrinsics: not captured yet")
                    elif calib_phase_state["phase"] == "extrinsics":
                        if pose_result is not None and cam_id in pose_result["poses"]:
                            extra.append(f"extrinsics: ✓ connected ({pose_graph.sample_count_for(cam_id)} obs)")
                        else:
                            best = pose_graph.best_pair_count_for(cam_id)
                            need = cfg["extrinsics"]["min_edge_observations"]
                            extra.append(
                                "extrinsics: no shared board view yet" if best == 0
                                else f"extrinsics: {best}/{need} shared views with best-paired camera"
                            )
                operational = []
                if flash_detect_state["armed"]:
                    flash_line = "FLASH: visible" if flash_detect_state["seen"][cam_id] else "FLASH: not in view"
                    extra.append(flash_line)
                    operational.append(flash_line)
                if cam_id in undistort_missing_intrinsics:
                    undistort_line = "undistort: enabled but no usable intrinsics -- showing raw frame"
                    extra.append(undistort_line)
                    operational.append(undistort_line)
                per_cam_operational[cam_id] = operational
                # Latest live board detection's reprojection error (px + mm),
                # not the intrinsics calibration's aggregate RMS -- see
                # state.reproj_error above for that (shown separately when
                # intrinsics_locked). mm uses the pinhole small-angle
                # approximation (mm = px * depth / focal_length_px), reusing
                # the K/t already solved for this detection.
                if state.last_edge_pose is not None:
                    _, _, edge_t, edge_err = state.last_edge_pose
                    f_avg = (state.K[0, 0] + state.K[1, 1]) / 2.0
                    mm_err = edge_err * (edge_t[2] / f_avg) * 1000.0
                    reproj = f"{edge_err:.3f} px ({mm_err:.2f} mm)"
                else:
                    reproj = "n/a"
                # Coverage/calibration-RMS numbers, spelled out explicitly
                # (not just left to the blue/orange coverage-overlay tint)
                # since during a focused single-camera intrinsics capture
                # the operator needs to read these at a glance against their
                # actual pass/fail thresholds -- not just "some of it is
                # blue". state.reproj_error is the RMS over ALL accepted
                # samples from the last cv2.calibrateCamera pass (what
                # actually gates `converged`), distinct from the live
                # single-frame PnP error below (board reproj error), which
                # only reflects THIS tick's detection and says nothing about
                # overall calibration quality.
                coverage_pct = state.coverage.ratio() * 100.0
                calib_reproj = (
                    f"{state.reproj_error:.3f} px" if state.reproj_error != float("inf")
                    else "n/a (not calibrated yet)"
                )
                per_cam_metrics[cam_id] = [
                    f"samples for int: {state.sample_count()}/{ic['max_samples_per_camera']}",
                    f"coverage: {coverage_pct:.0f}% (need ≥ {ic['coverage_ratio_threshold'] * 100:.0f}%)",
                    f"calibration reproj error: {calib_reproj} (need ≤ {ic['reproj_error_threshold_px']:.2f}px)",
                    f"live board reproj error: {reproj}",
                    f"samples for ext: {pose_graph.sample_count_for(cam_id)}",
                    f"converged: {'yes' if state.converged else 'no'}",
                    *extra,
                ]

            if flash_detect_state["armed"]:
                n_seen = sum(1 for v in flash_detect_state["seen"].values() if v)
                flash_detect_status_md.content = f"_Flash visible in {n_seen}/{len(cam_ids)} cameras._"
            else:
                flash_detect_status_md.content = ""

            # Step 1/2 button+status refresh -- every tick, since convergence
            # (RecalibrationWorker) and pose-graph solves (PoseGraphWorker)
            # both happen asynchronously off this thread, not only right
            # after a click.
            all_intrinsics_done = all(
                calib_states[c].converged or calib_states[c].intrinsics_locked for c in cam_ids
            )
            for cam_id in cam_ids:
                state = calib_states[cam_id]
                btn = intrinsics_btns[cam_id]
                if not sessions[cam_id].connected:
                    btn.disabled = True
                    btn.label, btn.color = f"{cam_id}: not connected", "gray"
                elif calib_phase_state["active_cam"] == cam_id:
                    btn.disabled = False
                    btn.label, btn.color = f"Capturing {cam_id}... (click to stop)", "red"
                elif state.converged or state.intrinsics_locked:
                    btn.disabled = False
                    btn.label, btn.color = f"{cam_id}: done (click to redo)", "green"
                else:
                    btn.disabled = False
                    btn.label, btn.color = f"Capture intrinsics: {cam_id}", "gray"

                # "Accept anyway" -- only ever useful while this camera has a
                # real (finite) reproj_error but hasn't converged/cached yet;
                # once it's done (real convergence OR a prior accept-anyway)
                # there's nothing left to override.
                accept_btn = accept_anyway_btns[cam_id]
                can_accept = (
                    not state.intrinsics_locked and not state.converged
                    and state.reproj_error != float("inf")
                )
                accept_btn.disabled = not can_accept
                accept_btn.label = (
                    f"Accept {cam_id} anyway ({state.reproj_error:.2f}px)" if can_accept
                    else f"Accept {cam_id} intrinsics anyway"
                )

            start_extrinsics_btn.disabled = not all_intrinsics_done or len(cam_ids) < 2
            start_extrinsics_btn.label = (
                "Restart extrinsics capture" if calib_phase_state["phase"] == "extrinsics"
                else "Capture extrinsics"
            )
            if not all_intrinsics_done:
                not_done = [c for c in cam_ids if not (calib_states[c].converged or calib_states[c].intrinsics_locked)]
                start_extrinsics_status_md.content = f"_Still need intrinsics: {', '.join(not_done)}_"
            elif len(cam_ids) < 2:
                start_extrinsics_status_md.content = "_Need at least 2 cameras for extrinsics._"
            else:
                start_extrinsics_status_md.content = ""

            if calib_phase_state["phase"] == "extrinsics":
                if pose_result is None:
                    extrinsics_guidance_md.content = (
                        "_Move the board so pairs of cameras can see it together..._"
                    )
                else:
                    n_connected = len(pose_result["poses"])
                    lines = [f"**{n_connected}/{len(cam_ids)} cameras connected.**"]
                    if pose_result["excluded"]:
                        lines.append(f"Not yet connected: {', '.join(sorted(pose_result['excluded']))}")
                    good = not pose_result["excluded"] and len(pose_result["components"]) == 1
                    if pose_result["chain_consistency_error_deg"]:
                        max_err = max(pose_result["chain_consistency_error_deg"].values())
                        threshold = cfg["extrinsics"]["naive_chain_error_threshold_deg"]
                        lines.append(
                            f"Max chain-consistency error: {max_err:.2f}° (target ≤ {threshold:.1f}°)"
                        )
                        good = good and max_err <= threshold
                    lines.append(
                        "✅ Extrinsics look good -- ready to Save." if good
                        else "⚠️ Keep moving the board for more overlap."
                    )
                    extrinsics_guidance_md.content = "\n\n".join(lines)
            else:
                extrinsics_guidance_md.content = ""

            if calib_phase_state["active_cam"] is not None:
                intrinsics_preview_status_md.content = "\n\n".join(
                    per_cam_metrics.get(calib_phase_state["active_cam"], [])
                )

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
        # Mailbox queues, no backlog to drain (unlike postproc below) -- each
        # worker.stop() is a stop-event set + a short bounded join.
        for cam_id, worker in calib_workers.items():
            worker.stop()
            if worker.is_alive():
                print(f"[Stop] {cam_id}'s detection worker didn't stop in time -- abandoning it.")
        for cam_id, worker in recalib_workers.items():
            worker.stop()
            if worker.is_alive():
                print(f"[Stop] {cam_id}'s recalibration worker didn't stop in time -- abandoning it.")
        pose_graph_worker.stop()
        if pose_graph_worker.is_alive():
            print("[Stop] The pose-graph worker didn't stop in time -- abandoning it.")
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
