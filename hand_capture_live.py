"""Live viser app: either load a calibration saved by calibrate.py or run one live,
then triangulate MediaPipe hand landmarks from all cameras into a live 3D skeleton.

Reuses calibrate.py's camera pipeline / calibration pieces and hand_multiview.py's
triangulation/visualization helpers directly rather than reimplementing them -- the
only genuinely new logic here is the viser-driven startup picker, the freeze
transition from "calibrating" to "tracking", and the live per-frame triangulation loop.

Camera poses (and intrinsics) are frozen the moment tracking starts, matching the
"cameras are static for the whole session" assumption already used throughout this
project (see calibrate.py's module docstring and multiview_hand_triangulation.ipynb's
limitations section) -- there is no per-frame re-estimation of camera poses here.
"""
import argparse
import glob
import os
import queue
import threading
import time

import cv2
import mediapipe as mp

try:
    import depthai as dai
except ImportError:
    dai = None

import calibrate as cal
import hand_multiview as hmv

# A hand-tracking button that enables the instant ≥1 camera is posed would be
# meaningless (triangulation needs ≥2 views); "a bit of calibration, so the cameras
# are in place" (the user's own framing) is read here as "≥2 cameras have a resolved
# extrinsic pose", not full intrinsic convergence.
MIN_POSED_CAMERAS_FOR_TRACKING = 2

# MediaPipe's own handedness confidence score, not the label -- see hand_multiview.HAND_LABEL.
TRACKING_MIN_CONFIDENCE = 0.5

# Looser than calibrate.py's 80ms simultaneity tolerance: undistort (4K remap) + MediaPipe
# Hands.process per camera is slower than ChArUco detection, so a tighter window would
# starve triangulation of any usable pairs. Still tight enough that a fast-moving hand
# doesn't get triangulated from geometrically-inconsistent stale views.
TRACKING_SIMULTANEITY_TOLERANCE_S = 0.15


def read_stdin_commands(cmd_queue, stop_event):
    while not stop_event.is_set():
        try:
            line = input()
        except EOFError:
            break
        cmd_queue.put(line.strip().lower())


# ==============================================================================
# Startup picker (viser, not a CLI flag)
# ==============================================================================
def list_calibrations(cfg):
    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    return sorted(glob.glob(os.path.join(out_dir, "*.json")), reverse=True)


def run_startup_picker(viser_mgr, cfg, app_status):
    """Blocks (polling) until the user picks a saved calibration or chooses to start
    a new one. Returns a file path, or None for "start new calibration". Both buttons
    disable themselves the instant they're clicked (so double-clicks can't fire twice
    and the user gets immediate feedback), well before the slow camera-startup work
    that follows actually begins.
    """
    files = list_calibrations(cfg)
    labels = [os.path.basename(f) for f in files] or ["(none saved yet)"]
    choice = {"decided": False, "path": None}

    with viser_mgr.server.gui.add_folder("Startup"):
        viser_mgr.server.gui.add_markdown(
            "Load a saved calibration below, or start a new live calibration."
        )
        dropdown = viser_mgr.server.gui.add_dropdown("Load calibration", options=labels)
        load_button = viser_mgr.server.gui.add_button("Load selected", disabled=not files)
        fresh_button = viser_mgr.server.gui.add_button("Start new calibration")

    @load_button.on_click
    def _(_):
        load_button.disabled = True
        fresh_button.disabled = True
        app_status.content = "**Status:** loading calibration and connecting cameras..."
        choice["path"] = files[labels.index(dropdown.value)]
        choice["decided"] = True

    @fresh_button.on_click
    def _(_):
        load_button.disabled = True
        fresh_button.disabled = True
        app_status.content = "**Status:** starting cameras..."
        choice["decided"] = True

    print("[Startup] Waiting for a choice in viser: load a saved calibration, or start a new one...")
    while not choice["decided"]:
        time.sleep(0.05)
    return choice["path"]


def open_all_camera_sessions(cfg, app_status):
    """Opens every currently-connected OAK camera immediately at launch, before the
    user has made any startup-picker choice -- there's no reason to make cameras
    (which take a few seconds each to connect/start their pipeline) wait on a GUI
    click, since both startup paths need them regardless of which one is chosen.
    """
    device_infos = dai.Device.getAllAvailableDevices()
    sessions = {}
    for i, info in enumerate(device_infos):
        cam_id = f"cam{i}"
        app_status.content = f"**Status:** starting cameras... ({cam_id})"
        print(f"Starting {cam_id} ({info.deviceId})...")
        try:
            sessions[cam_id] = cal.CalibCameraSession(cam_id, info, cfg)
        except Exception as exc:
            print(f"[Error] Failed to start {cam_id}: {exc}")
    return sessions


def match_open_sessions_by_device_id(all_sessions, loaded_cameras):
    """loaded_cameras: dict[cam_id] -> {..., "device_id"}. Matches saved cameras to
    already-open sessions by device_id -- camera enumeration order is not guaranteed
    stable across process restarts, so assuming the same cam0/cam1/... order would
    silently mismatch K/dist/pose to the wrong physical camera. Returns
    (matched: dict[loaded_cam_id -> session], missing: list[loaded_cam_id]).
    """
    by_device_id = {session.device_id: session for session in all_sessions.values()}
    matched, missing = {}, []
    for cam_id, data in loaded_cameras.items():
        session = by_device_id.get(data["device_id"])
        if session is None:
            missing.append(cam_id)
        else:
            matched[cam_id] = session
    return matched, missing


def build_tracking_state(cameras):
    """cameras: dict[cam_id] -> {"K", "dist", "width", "height", "R", "t"}.
    Returns dict[cam_id] -> same plus undistort maps and a projection matrix, ready
    for live per-frame undistortion + triangulation.
    """
    state = {}
    for cam_id, c in cameras.items():
        map1, map2 = hmv.build_undistort_maps(c["K"], c["dist"], c["width"], c["height"])
        state[cam_id] = {
            "K": c["K"], "dist": c["dist"], "width": c["width"], "height": c["height"],
            "R": c["R"], "t": c["t"],
            "map1": map1, "map2": map2,
            "P": hmv.build_projection_matrix(c["K"], c["R"], c["t"]),
        }
    return state


def show_frozen_poses(viser_mgr, tracking_state):
    for cam_id, s in tracking_state.items():
        T_cam_to_world = cal.invert_T(cal.rt_to_T(s["R"], s["t"]))
        viser_mgr.update_camera_pose(cam_id, T_cam_to_world, s["K"], s["width"], s["height"])


# ==============================================================================
# Calibration mode -- mirrors calibrate.py's main loop body, reusing its pieces
# directly (CalibCameraSession, CameraCalibState, PoseGraph, ViserManager,
# detect_charuco, process_detection, maybe_recalibrate, try_reconnect, save_output).
# The extra behavior here is the per-tick tracking_ready_cb hook that gates and
# handles the "Start hand tracking" freeze transition.
# ==============================================================================
def run_calibration_mode(cfg, viser_mgr, sessions, cmd_queue, tracking_ready_cb, app_status, deadline=None):
    """`sessions` must already be open (see open_all_camera_sessions) -- cameras start
    up immediately at launch, independent of the startup-picker choice.
    """
    _board, detector, board_points_3d = cal.build_board(cfg)
    cam_ids = list(sessions.keys())
    image_size = (cfg["camera"]["record_width"], cfg["camera"]["record_height"])
    calib_states = {cam_id: cal.CameraCalibState(cam_id, image_size, cfg) for cam_id in cam_ids}

    viser_mgr.add_global_controls(cmd_queue)
    world_align = {"pending": False, "R": None}

    def request_world_alignment():
        world_align["pending"] = True
        print("[Align] Waiting for a fresh board detection to set the world's down direction...")

    viser_mgr.add_world_alignment_button(request_world_alignment)

    app_status.content = (
        f"**Status:** calibrating -- waiting for at least {MIN_POSED_CAMERAS_FOR_TRACKING} "
        f"camera(s) to be posed"
    )

    pose_graph = cal.PoseGraph(cfg)
    runtime_cfg = cfg["runtime"]
    tol_s = cfg["extrinsics"]["simultaneity_tolerance_ms"] / 1000.0
    last_status_print = time.monotonic()
    last_reconnect_attempt = {cam_id: 0.0 for cam_id in cam_ids}
    pose_result = None
    result = (None, None)

    try:
        while True:
            if deadline is not None and time.monotonic() > deadline:
                print("[Duration] elapsed during calibration mode -- stopping.")
                break

            while not cmd_queue.empty():
                cmd = cmd_queue.get()
                if cmd in ("q", "quit", "exit"):
                    raise KeyboardInterrupt
                if cmd == "save":
                    device_ids = {c: sessions[c].device_id for c in cam_ids}
                    cal.save_output(cam_ids, calib_states, pose_result, cfg, device_ids)
                elif cmd.startswith("reset"):
                    parts = cmd.split()
                    target = parts[1] if len(parts) > 1 else "all"
                    for cam_id in (cam_ids if target == "all" else [target]):
                        if cam_id in calib_states:
                            calib_states[cam_id] = cal.CameraCalibState(cam_id, image_size, cfg)
                            print(f"[Reset] {cam_id} calibration state cleared.")

            now = time.monotonic()

            # Fast pass: fetch/decode every camera's latest frame first (see module
            # docstring / calibrate.py's fix -- timestamps taken after slow per-camera
            # work silently break cross-camera time-matching).
            fresh_frames = {}
            for cam_id in cam_ids:
                session, state = sessions[cam_id], calib_states[cam_id]
                if not session.connected:
                    if now - last_reconnect_attempt[cam_id] >= runtime_cfg["reconnect_retry_interval_s"]:
                        last_reconnect_attempt[cam_id] = now
                        cal.try_reconnect(session, cam_id)
                        state.connected = session.connected
                    continue
                try:
                    got_frame = session.poll()
                except Exception as exc:
                    print(f"[Error] {cam_id} disconnected: {exc}")
                    session.close()
                    state.connected = False
                    continue
                if session.last_frame_preview is not None:
                    viser_mgr.update_thumbnail(cam_id, session.last_frame_preview, state.coverage)
                if got_frame and session.last_frame_full is not None:
                    fresh_frames[cam_id] = (session.last_frame_full, session.last_frame_full_ts)

            # Slow pass: ChArUco detection + quality gates + PnP.
            for cam_id, (frame, frame_ts) in fresh_frames.items():
                state = calib_states[cam_id]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detection = cal.detect_charuco(detector, gray)
                if detection is None:
                    # No board seen this tick -- hide it rather than leaving this
                    # camera's last-ever sighting stuck on screen forever.
                    viser_mgr.hide_board_pose(cam_id)
                    continue
                corners2d, ids = detection

                pose, accepted = cal.process_detection(state, corners2d, ids, board_points_3d, gray, cfg)
                if accepted:
                    cal.maybe_recalibrate(state, cfg)
                if pose is None:
                    viser_mgr.hide_board_pose(cam_id)
                    continue
                R, t, err = pose
                state.last_detection_ts = frame_ts
                state.last_edge_pose = (frame_ts, R, t, err)

                if pose_result is not None and cam_id in pose_result["poses"]:
                    T_cam_to_world = cal.invert_T(pose_result["poses"][cam_id])
                    viser_mgr.update_board_pose(cam_id, T_cam_to_world @ cal.rt_to_T(R, t))
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
                    # Retried every tick (not just the tick of the click) until some
                    # camera actually has a fresh, posed board detection to align to.
                    now_wall = time.time()
                    for cam_id in cam_ids:
                        edge = calib_states[cam_id].last_edge_pose
                        if edge is None or cam_id not in pose_result["poses"]:
                            continue
                        edge_ts, R_bc, t_bc, _err = edge
                        if now_wall - edge_ts > 2.0:
                            continue
                        T_cam_to_world = cal.invert_T(pose_result["poses"][cam_id])
                        world_align["R"] = cal.compute_board_world_alignment(T_cam_to_world, R_bc, t_bc)
                        world_align["pending"] = False
                        print(f"[Align] World down direction set from {cam_id}'s board detection.")
                        break
                cal.apply_world_alignment(pose_result, world_align["R"])

                for cam_id in cam_ids:
                    state = calib_states[cam_id]
                    if cam_id in pose_result["poses"]:
                        T_cam_to_world = cal.invert_T(pose_result["poses"][cam_id])
                        viser_mgr.update_camera_pose(cam_id, T_cam_to_world, state.K, *state.image_size)
                    else:
                        viser_mgr.mark_pose_unknown(cam_id)

            for cam_id in cam_ids:
                viser_mgr.update_status(cam_id, calib_states[cam_id])

            if now - last_status_print >= runtime_cfg["status_print_interval_s"]:
                last_status_print = now
                posed = len(pose_result["poses"]) if pose_result else 0
                parts = [
                    f"{c}: {calib_states[c].sample_count()} samples, "
                    f"reproj={calib_states[c].reproj_error:.2f}px"
                    for c in cam_ids
                ]
                print(f"[Status] {posed}/{len(cam_ids)} posed | " + " | ".join(parts))

            frozen = tracking_ready_cb(pose_result, cam_ids, calib_states)
            if frozen is not None:
                result = (sessions, frozen)
                break

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n[Stop] Interrupted by user.")

    finally:
        if result[0] is None:
            for session in sessions.values():
                session.close()

    return result


# ==============================================================================
# Tracking mode -- live per-camera MediaPipe + N-view triangulation
# ==============================================================================
def run_tracking_mode(viser_mgr, sessions, tracking_state, cmd_queue, app_status, deadline=None):
    cam_ids = list(tracking_state.keys())
    app_status.content = f"**Status:** starting hand tracking ({len(cam_ids)} camera(s))..."
    hands_by_cam = {
        cam_id: mp.solutions.hands.Hands(
            max_num_hands=1, min_detection_confidence=0.2, min_tracking_confidence=0.5,
        )
        for cam_id in cam_ids
    }
    latest_obs = {cam_id: None for cam_id in cam_ids}  # (landmarks, confidence, ts, w, h)

    show_frozen_poses(viser_mgr, tracking_state)
    app_status.content = f"**Status:** tracking active ({len(cam_ids)} camera(s))"
    print(f"[Tracking] Live hand triangulation started with {len(cam_ids)} camera(s).")

    try:
        while True:
            if deadline is not None and time.monotonic() > deadline:
                print("[Duration] elapsed during tracking mode -- stopping.")
                break

            while not cmd_queue.empty():
                cmd = cmd_queue.get()
                if cmd in ("q", "quit", "exit"):
                    raise KeyboardInterrupt

            # Fast pass: fetch/decode every camera's latest frame first, same
            # rationale as calibration mode -- and more important here, since a
            # moving hand needs genuinely simultaneous views to triangulate cleanly.
            fresh_frames = {}
            for cam_id in cam_ids:
                session = sessions[cam_id]
                try:
                    got = session.poll()
                except Exception as exc:
                    print(f"[Error] {cam_id} disconnected during tracking: {exc}")
                    continue
                if session.last_frame_preview is not None:
                    viser_mgr.update_thumbnail_raw(cam_id, session.last_frame_preview)
                if got and session.last_frame_full is not None:
                    fresh_frames[cam_id] = (session.last_frame_full, session.last_frame_full_ts)

            # Slow pass: undistort + MediaPipe.
            for cam_id, (frame, ts) in fresh_frames.items():
                s = tracking_state[cam_id]
                undistorted = hmv.undistort_fast(frame, s["map1"], s["map2"])
                landmarks, confidence = hmv.extract_single_frame_landmarks(hands_by_cam[cam_id], undistorted)
                if landmarks is not None and confidence >= TRACKING_MIN_CONFIDENCE:
                    latest_obs[cam_id] = (landmarks, confidence, ts, s["width"], s["height"])

            # Only cameras whose latest observation is close in time to the newest one
            # contribute to this tick's triangulation -- a stale view of a moving hand
            # is geometrically inconsistent with a fresh one.
            valid = {c: o for c, o in latest_obs.items() if o is not None}
            frame_points = {}
            if len(valid) >= 2:
                newest_ts = max(o[2] for o in valid.values())
                points_2d_by_landmark = {str(i): {} for i in range(21)}
                for cam_id, (landmarks, _confidence, ts, w, h) in valid.items():
                    if newest_ts - ts > TRACKING_SIMULTANEITY_TOLERANCE_S:
                        continue
                    pixel_xy = hmv.landmark_dict_to_pixel_xy(landmarks, w, h)
                    for lm_id, xy in pixel_xy.items():
                        points_2d_by_landmark[str(lm_id)][cam_id] = tuple(xy)

                for lm_id, points_2d in points_2d_by_landmark.items():
                    if len(points_2d) < 2:
                        continue
                    projection_matrices = {c: tracking_state[c]["P"] for c in points_2d}
                    xyz = hmv.triangulate_dlt(points_2d, projection_matrices)
                    if xyz is not None:
                        frame_points[lm_id] = xyz.tolist()

            hmv.add_skeleton_frame(viser_mgr.server, frame_points)
            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n[Stop] Tracking interrupted by user.")

    finally:
        for hands in hands_by_cam.values():
            hands.close()
        for session in sessions.values():
            session.close()


# ==============================================================================
# main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Live multi-camera MediaPipe hand triangulation")
    parser.add_argument("--config", default=cal.CONFIG_PATH_DEFAULT, help="Path to calibrate.py's YAML config")
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Optional auto-stop after N seconds, applied within each mode (mainly for headless/automated runs)",
    )
    args = parser.parse_args()

    if dai is None:
        print("[Error] depthai is not installed in this environment.")
        return

    cfg = cal.load_config(args.config)
    viser_mgr = cal.ViserManager(cfg)
    viser_mgr.add_camera_settings_panel()
    app_status = viser_mgr.server.gui.add_markdown("**Status:** starting cameras...")
    print(f"[Viser] http://localhost:{viser_mgr.server.get_port()}")

    cmd_queue = queue.Queue()
    stop_event = threading.Event()
    threading.Thread(target=read_stdin_commands, args=(cmd_queue, stop_event), daemon=True).start()

    start_time = time.monotonic()
    deadline = start_time + args.duration if args.duration is not None else None

    # Cameras start immediately -- they don't need to wait on the startup-picker
    # choice below, since either path (load or fresh-calibrate) needs them anyway.
    all_sessions = open_all_camera_sessions(cfg, app_status)
    if not all_sessions:
        app_status.content = "**Status:** error -- no OAK devices discovered"
        print("[Error] No OAK devices discovered.")
        return
    app_status.content = f"**Status:** {len(all_sessions)} camera(s) ready -- choose a calibration source"

    chosen_path = run_startup_picker(viser_mgr, cfg, app_status)

    if chosen_path is not None:
        print(f"[Startup] Loading calibration from {chosen_path}")
        loaded = cal.load_calibration_output(chosen_path)
        sessions, missing = match_open_sessions_by_device_id(all_sessions, loaded)
        if missing:
            print(f"[Warning] Saved cameras not found among connected devices, excluded: {missing}")
        for session in all_sessions.values():
            if session not in sessions.values():
                session.close()
        if len(sessions) < MIN_POSED_CAMERAS_FOR_TRACKING:
            app_status.content = (
                f"**Status:** error -- only {len(sessions)} camera(s) matched, "
                f"need at least {MIN_POSED_CAMERAS_FOR_TRACKING}"
            )
            print(f"[Error] Only {len(sessions)} camera(s) matched -- need at least "
                  f"{MIN_POSED_CAMERAS_FOR_TRACKING} to triangulate.")
            return
        tracking_state = build_tracking_state({cam_id: loaded[cam_id] for cam_id in sessions})
    else:
        with viser_mgr.server.gui.add_folder("Tracking"):
            start_button = viser_mgr.server.gui.add_button("Start hand tracking", disabled=True)
        tracking_flag = {"start": False}

        @start_button.on_click
        def _(_):
            # Instant feedback on click, well before the freeze actually happens on
            # the next tick of the calibration loop.
            start_button.disabled = True
            app_status.content = "**Status:** freezing camera poses, starting hand tracking..."
            tracking_flag["start"] = True

        def tracking_ready_cb(pose_result, cam_ids, calib_states):
            posed = pose_result["poses"] if pose_result else {}
            if not tracking_flag["start"]:
                start_button.disabled = len(posed) < MIN_POSED_CAMERAS_FOR_TRACKING
            if len(posed) < MIN_POSED_CAMERAS_FOR_TRACKING or not tracking_flag["start"]:
                return None
            cameras = {
                cam_id: {
                    "K": calib_states[cam_id].K,
                    "dist": calib_states[cam_id].dist,
                    "width": calib_states[cam_id].image_size[0],
                    "height": calib_states[cam_id].image_size[1],
                    "R": cal.T_to_rt(posed[cam_id])[0],
                    "t": cal.T_to_rt(posed[cam_id])[1],
                }
                for cam_id in posed
            }
            return build_tracking_state(cameras)

        sessions, tracking_state = run_calibration_mode(
            cfg, viser_mgr, all_sessions, cmd_queue, tracking_ready_cb, app_status, deadline
        )
        if sessions is None:
            print("[Stop] Exiting without starting hand tracking.")
            stop_event.set()
            return

    run_tracking_mode(viser_mgr, sessions, tracking_state, cmd_queue, app_status, deadline)
    stop_event.set()


if __name__ == "__main__":
    main()
