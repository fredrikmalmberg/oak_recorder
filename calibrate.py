"""Multi-camera live ChArUco calibration tool (intrinsics + extrinsics) with a
real-time viser viewer. Discovers however many OAK cameras are plugged in
(mirroring capture.py's camera enumeration/pipeline pattern), auto-collects
calibration samples per camera, and builds a pose graph across cameras from
frames where two or more of them see the board at (approximately) the same
time -- cameras are free-running and not hardware-synced, so "same time" is a
configurable tolerance rather than a shared frame index.

Open assumptions (spec section 12) -- flagged here since they were not fully
specified by the user and may need revisiting:
  - Reference camera is auto-selected as the most-connected node in the pose
    graph (ties broken by total edge observation count).
  - Quality-gate threshold defaults (corner count, blur variance, coverage
    grid resolution, angle/pose diversity) are guesses tuned for a 6x4
    40mm/30mm board at 4K; see `DEFAULT_CONFIG` below and calibrate_config.yaml.
  - "Converged" = mean reprojection error under a threshold AND coverage grid
    sufficiently filled (see `intrinsics` config section).
  - Simultaneity tolerance for pairing free-running cameras' frames into an
    extrinsic edge defaults to 80ms; see `extrinsics.simultaneity_tolerance_ms`.
  - Live viser thumbnail downsample factor is a config value
    (`camera.display_width/height`), default 480x270.
  - The angle-diversity gate (spec 6, bullet 3) and the spatial/pose-distance
    gate (spec 6, bullet 5) are consolidated into a single novelty check: a
    sample is rejected only if it is close to some already-accepted sample in
    *both* translation and rotation (see `passes_novelty_gate`).
  - Pose-graph "naive chaining reprojection error" (spec section 5b) is
    measured as chain-consistency disagreement across redundant (cycle)
    edges, not true per-corner pixel reprojection -- with no cycles in a
    component there is nothing for a chain to disagree with, so its error is
    definitionally zero and optimization would not change anything.
"""
import argparse
import json
import math
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np
import yaml

try:
    import depthai as dai
except ImportError:
    dai = None

import networkx as nx
import viser
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

CONFIG_PATH_DEFAULT = "calibrate_config.yaml"

DEFAULT_CONFIG = {
    "board": {
        "squares_x": 6,
        "squares_y": 4,
        "square_size_m": 0.040,
        "marker_size_m": 0.030,
        "aruco_dict": "DICT_4X4_50",
    },
    "camera": {
        "record_width": 3840,
        "record_height": 2160,
        "display_width": 480,
        "display_height": 270,
        "fps": 15,
        "mjpeg_quality": 90,
        "shutter_us": 8000,
        "iso": 400,
        "wb_k": 4500,
    },
    "quality_gates": {
        "min_corners": 8,
        "blur_laplacian_var_min": 30.0,
        "coverage_grid_rows": 4,
        "coverage_grid_cols": 6,
        "min_coverage_cells_per_frame": 4,
        "min_pose_translation_diversity_m": 0.05,
        "min_pose_angle_diversity_deg": 8.0,
    },
    "intrinsics": {
        "reproj_error_threshold_px": 0.6,
        "coverage_ratio_threshold": 0.8,
        "min_samples_before_calibrate": 8,
        "recalibrate_every_n_samples": 3,
        "max_samples_per_camera": 150,
    },
    "extrinsics": {
        "simultaneity_tolerance_ms": 80.0,
        "min_edge_observations": 5,
        "use_pose_graph_optimization": True,
        "naive_chain_error_threshold_deg": 1.0,
    },
    "runtime": {
        "auto_stop": False,
        "status_print_interval_s": 3.0,
        "reconnect_retry_interval_s": 5.0,
        "no_detection_warning_s": 20.0,
    },
    "output": {
        "dir": "calibrations",
        "save_raw_images": False,
    },
    "viser": {
        "port": 8080,
    },
}


def _deep_merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(DEFAULT_CONFIG, f, sort_keys=False)
        print(f"[Config] No config found -- wrote defaults to {path}")
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    _deep_merge(cfg, user_cfg)
    return cfg


# ==============================================================================
# ChArUco board
# ==============================================================================
def build_board(cfg):
    b = cfg["board"]
    dict_id = getattr(cv2.aruco, b["aruco_dict"], None)
    if dict_id is None:
        options = [n for n in dir(cv2.aruco) if n.startswith("DICT_4X4_")]
        raise ValueError(f"Unknown aruco_dict '{b['aruco_dict']}'. Try one of: {options}")
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard(
        (b["squares_x"], b["squares_y"]), b["square_size_m"], b["marker_size_m"], aruco_dict,
    )
    detector_params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.CharucoDetector(board, detectorParams=detector_params)
    board_points_3d = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    return board, detector, board_points_3d


def detect_charuco(detector, gray):
    """Returns (corners2d[N,2], ids[N]) or None. Matches this repo's installed
    OpenCV (5.0.0), where CharucoDetector.detectBoard returns a plain 4-tuple
    (charucoCorners, charucoIds, markerCorners, markerIds) rather than a
    result object -- verified directly against the installed build.
    """
    charuco_corners, charuco_ids, _marker_corners, _marker_ids = detector.detectBoard(gray)
    if charuco_ids is None or len(charuco_ids) == 0:
        return None
    ids = np.asarray(charuco_ids).reshape(-1)
    corners2d = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
    return corners2d, ids


# ==============================================================================
# Rigid-transform helpers (4x4 homogeneous matrices throughout)
# ==============================================================================
def rt_to_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).ravel()
    return T


def T_to_rt(T):
    return T[:3, :3].copy(), T[:3, 3].copy()


def invert_T(T):
    R, t = T[:3, :3], T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def R_to_wxyz(R):
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


# ==============================================================================
# World alignment ("set down direction"): re-bases the whole scene so that a
# board placed flat on the floor (camera looking down at it) defines gravity.
#
# viser's default up direction is +Z (like Blender/ROS/most robotics tooling --
# not +Y as in some game engines/OpenGL-style conventions, which is a common
# assumption to get wrong here). We align the board's own axes directly to the
# world axes -- local Z (solvePnP's out-of-the-face normal, which points toward
# the camera, i.e. upward, when the camera looks down at a floor-flat board) to
# world +Z, local X/Y to world X/Y -- rather than calling viser's
# set_up_direction, which only affects camera controls/lighting, not the actual
# geometry of already-placed frustums/points.
# ==============================================================================
def compute_board_world_alignment(T_cam_to_world, R_board_to_cam, t_board_to_cam):
    """Returns the board's rotation in the current (pre-alignment) world frame --
    see apply_world_alignment for how this re-bases every stored camera pose so
    that rotation becomes the identity (board's own axes == world axes).
    """
    T_board_to_world = T_cam_to_world @ rt_to_T(R_board_to_cam, t_board_to_cam)
    R_board_to_world, _ = T_to_rt(T_board_to_world)
    return R_board_to_world


def apply_world_alignment(pose_result, alignment_R):
    """Mutates pose_result["poses"] in place. Call every tick on the fresh,
    unaligned PoseGraph.solve() output -- alignment_R (the board's world rotation
    at the moment the user confirmed it was flat, from compute_board_world_alignment)
    is a fixed value from here on, not something the pose graph itself knows about.

    Derivation: post-multiplying each stored world-to-camera transform T_wc by
    [alignment_R | 0] is equivalent to rotating the world frame itself by
    alignment_R^-1 -- i.e. exactly cancels the board's own rotation, making the
    board's local axes equal the (new) world axes.
    """
    if pose_result is None or alignment_R is None:
        return
    T_realign = rt_to_T(alignment_R, np.zeros(3))
    for cam_id in pose_result["poses"]:
        pose_result["poses"][cam_id] = pose_result["poses"][cam_id] @ T_realign


# ==============================================================================
# Per-camera calibration state
# ==============================================================================
class CoverageGrid:
    def __init__(self, rows, cols, width, height):
        self.rows, self.cols = rows, cols
        self.width, self.height = width, height
        self.filled = np.zeros((rows, cols), dtype=bool)

    def cell_of(self, xy):
        col = min(self.cols - 1, max(0, int(xy[0] / self.width * self.cols)))
        row = min(self.rows - 1, max(0, int(xy[1] / self.height * self.rows)))
        return row, col

    def cells_touched(self, corners2d):
        return {self.cell_of(xy) for xy in corners2d}

    def new_cell_count(self, corners2d):
        return sum(1 for cell in self.cells_touched(corners2d) if not self.filled[cell])

    def mark(self, corners2d):
        for cell in self.cells_touched(corners2d):
            self.filled[cell] = True

    def ratio(self):
        return float(self.filled.mean())


class CameraCalibState:
    def __init__(self, cam_id, image_size, cfg):
        self.cam_id = cam_id
        self.image_size = image_size
        self.cfg = cfg
        qg = cfg["quality_gates"]
        self.coverage = CoverageGrid(qg["coverage_grid_rows"], qg["coverage_grid_cols"], *image_size)
        self.object_points = []
        self.image_points = []
        self.accepted_poses = []  # list of (rvec[3], tvec[3]) for the novelty gate
        w, h = image_size
        self.K = np.array([[w, 0.0, w / 2.0], [0.0, w, h / 2.0], [0.0, 0.0, 1.0]])
        self.dist = np.zeros(5)
        self.has_intrinsics_estimate = False
        self.reproj_error = float("inf")
        self.converged = False
        self.samples_since_calib = 0
        self.last_detection_ts = None
        self.last_edge_pose = None  # (ts, R, t, err) from the most recent PnP solve, for extrinsic edges
        self.connected = True
        self.last_warned_no_detection = False

    def sample_count(self):
        return len(self.object_points)


def laplacian_sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def solve_board_pose(object_points_3d, corners2d, K, dist):
    ok, rvec, tvec = cv2.solvePnP(object_points_3d, corners2d, K, dist)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    proj, _ = cv2.projectPoints(object_points_3d, rvec, tvec, K, dist)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - corners2d, axis=1).mean())
    return R, tvec.ravel(), err


def passes_novelty_gate(state, R, t, cfg):
    """Reject only true near-duplicates: a sample close to some already
    accepted sample in *both* translation and rotation. This also serves the
    angle-diversity gate (spec 6 bullet 3), since a sample whose tilt differs
    enough from every accepted sample will always pass here even at an
    already-visited position -- see the module docstring's open-assumptions note.
    """
    qg = cfg["quality_gates"]
    min_t = qg["min_pose_translation_diversity_m"]
    min_a = math.radians(qg["min_pose_angle_diversity_deg"])
    for R_prev, t_prev in state.accepted_poses:
        t_diff = float(np.linalg.norm(t - t_prev))
        R_diff = Rotation.from_matrix(R_prev.T @ R).magnitude()
        if t_diff < min_t and R_diff < min_a:
            return False
    return True


def process_detection(state, corners2d, ids, board_points_3d, gray, cfg):
    """Runs all quality gates; if the frame is accepted, adds it to the
    camera's intrinsic-calibration sample set and returns the board pose
    (R, t, reprojection error px) used for the extrinsics pose graph. Frames
    that fail the intrinsic-sample gates can still be returned for extrinsics
    purposes as long as they clear the corner-count and blur gates, since more
    simultaneous-detection data only helps the pose graph.
    """
    qg = cfg["quality_gates"]
    if len(ids) < qg["min_corners"]:
        return None, False

    if laplacian_sharpness(gray) < qg["blur_laplacian_var_min"]:
        return None, False

    obj = board_points_3d[ids]
    pose = solve_board_pose(obj, corners2d, state.K, state.dist)
    if pose is None:
        return None, False
    R, t, err = pose

    accepted = False
    if len(state.coverage.cells_touched(corners2d)) >= qg["min_coverage_cells_per_frame"]:
        if passes_novelty_gate(state, R, t, cfg):
            ic = cfg["intrinsics"]
            if state.sample_count() < ic["max_samples_per_camera"]:
                state.object_points.append(obj.astype(np.float32))
                state.image_points.append(corners2d.astype(np.float32))
                state.accepted_poses.append((R, t))
                state.coverage.mark(corners2d)
                state.samples_since_calib += 1
                accepted = True

    return (R, t, err), accepted


def maybe_recalibrate(state, cfg):
    ic = cfg["intrinsics"]
    if state.sample_count() < ic["min_samples_before_calibrate"]:
        return
    if state.samples_since_calib < ic["recalibrate_every_n_samples"]:
        return
    w, h = state.image_size
    flags = cv2.CALIB_USE_INTRINSIC_GUESS if state.has_intrinsics_estimate else 0
    try:
        ret, K, dist, _, _ = cv2.calibrateCamera(
            state.object_points, state.image_points, (w, h),
            state.K.copy(), state.dist.copy(), flags=flags,
        )
    except cv2.error as exc:
        print(f"[Calib] {state.cam_id}: calibrateCamera failed ({exc}); keeping previous estimate.")
        state.samples_since_calib = 0
        return
    state.K, state.dist, state.reproj_error = K, dist.ravel(), float(ret)
    state.has_intrinsics_estimate = True
    state.samples_since_calib = 0
    state.converged = (
        state.reproj_error <= ic["reproj_error_threshold_px"]
        and state.coverage.ratio() >= ic["coverage_ratio_threshold"]
    )


# ==============================================================================
# Camera capture layer -- replicates capture.py's pipeline pattern (manual
# exposure/white-balance Camera node -> hardware MJPEG encode for the full-res
# stream, small BGR888p stream for live preview) rather than importing it
# directly, since capture.py's pipeline builder reads its settings from
# module-level constants instead of a config dict.
# ==============================================================================
def build_calibration_pipeline(device, cfg):
    cam_cfg = cfg["camera"]
    pipeline = dai.Pipeline(device)

    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam.initialControl.setManualExposure(cam_cfg["shutter_us"], cam_cfg["iso"])
    cam.initialControl.setManualWhiteBalance(cam_cfg["wb_k"])

    out_full = cam.requestOutput(
        (cam_cfg["record_width"], cam_cfg["record_height"]),
        type=dai.ImgFrame.Type.NV12, fps=cam_cfg["fps"],
    )
    video_enc = pipeline.create(dai.node.VideoEncoder).build(
        out_full, frameRate=cam_cfg["fps"], profile=dai.VideoEncoderProperties.Profile.MJPEG,
    )
    video_enc.setQuality(cam_cfg["mjpeg_quality"])

    out_preview = cam.requestOutput(
        (cam_cfg["display_width"], cam_cfg["display_height"]),
        type=dai.ImgFrame.Type.BGR888p, fps=cam_cfg["fps"],
    )
    return pipeline, video_enc.out, out_preview


def _packet_bytes(msg):
    data = msg.getData()
    return data.tobytes() if hasattr(data, "tobytes") else bytes(data)


class CalibCameraSession:
    def __init__(self, cam_id, device_info, cfg):
        self.cam_id = cam_id
        self.cfg = cfg
        self.device_info = device_info
        self.device_id = device_info.deviceId
        self.connected = False
        self.last_frame_full = None
        self.last_frame_full_ts = None
        self.last_frame_preview = None
        self._connect()

    def _connect(self):
        self.device = dai.Device(self.device_info)
        self.pipeline, full_endpoint, preview_endpoint = build_calibration_pipeline(self.device, self.cfg)
        self.q_full = full_endpoint.createOutputQueue(maxSize=4, blocking=False)
        self.q_preview = preview_endpoint.createOutputQueue(maxSize=2, blocking=False)
        self.pipeline.start()
        self.connected = True

    def poll(self):
        """Fetches whatever frames are queued (non-blocking). Returns True if
        a new full-resolution frame arrived this call. Raises on a genuinely
        dead device so the caller can mark it disconnected and retry.
        """
        preview_msg = self.q_preview.tryGet()
        if preview_msg is not None:
            self.last_frame_preview = preview_msg.getCvFrame()

        full_msg = self.q_full.tryGet()
        if full_msg is None:
            return False
        frame = cv2.imdecode(np.frombuffer(_packet_bytes(full_msg), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return False
        self.last_frame_full = frame
        self.last_frame_full_ts = time.time()
        return True

    def close(self):
        self.connected = False
        try:
            self.pipeline.stop()
        except Exception:
            pass
        try:
            self.device.close()
        except Exception:
            pass


# ==============================================================================
# Extrinsics: pose graph across cameras
# ==============================================================================
class PoseGraph:
    """Nodes = cameras. An edge observation is derived whenever two cameras'
    most recent board detections fall within the simultaneity tolerance: each
    camera independently gives a board-in-camera pose (via its own current
    intrinsics), and the relative camera-to-camera transform is recovered by
    composing one against the inverse of the other. Repeated observations for
    the same camera pair are averaged (rotation via a mean over SO(3), per
    scipy's `Rotation.mean`) into a single edge with a confidence/error weight.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.observations = {}  # sorted (cam_a, cam_b) -> list[{"R","t","err"}]

    def add_observation(self, cam_a, R_a, t_a, err_a, cam_b, R_b, t_b, err_b):
        T_a = rt_to_T(R_a, t_a)  # board -> cam_a
        T_b = rt_to_T(R_b, t_b)  # board -> cam_b
        T_ab = T_b @ invert_T(T_a)  # cam_a -> cam_b

        key = tuple(sorted((cam_a, cam_b)))
        T_edge = T_ab if key == (cam_a, cam_b) else invert_T(T_ab)
        R_edge, t_edge = T_to_rt(T_edge)
        err = 0.5 * (err_a + err_b)
        self.observations.setdefault(key, []).append({"R": R_edge, "t": t_edge, "err": err})

    def aggregate_edges(self):
        min_obs = self.cfg["extrinsics"]["min_edge_observations"]
        edges = {}
        for key, obs in self.observations.items():
            if len(obs) < min_obs:
                continue
            R_mean = Rotation.from_matrix(np.stack([o["R"] for o in obs])).mean().as_matrix()
            t_mean = np.mean(np.stack([o["t"] for o in obs]), axis=0)
            edges[key] = {
                "R": R_mean, "t": t_mean,
                "mean_reproj_error_px": float(np.mean([o["err"] for o in obs])),
                "count": len(obs),
            }
        return edges

    def _edge_transform(self, edges, a, b):
        key = tuple(sorted((a, b)))
        e = edges[key]
        T = rt_to_T(e["R"], e["t"])
        return T if key == (a, b) else invert_T(T)

    def _chain_pose(self, g, edges, ref, cam):
        path = nx.shortest_path(g, ref, cam, weight="weight")
        T = np.eye(4)
        for a, b in zip(path[:-1], path[1:]):
            T = self._edge_transform(edges, a, b) @ T
        return T

    def _chain_consistency_error_deg(self, g, edges, poses, comp):
        """Only meaningful when the component has cycles (redundant edges) --
        a pure spanning tree has no alternative path to disagree with, so its
        chain-consistency error is definitionally zero.
        """
        if g.subgraph(comp).number_of_edges() <= len(comp) - 1:
            return 0.0
        angle_errs = []
        for a, b in g.subgraph(comp).edges():
            T_meas = self._edge_transform(edges, a, b)
            T_pred = poses[b] @ invert_T(poses[a])
            T_err = invert_T(T_meas) @ T_pred
            angle_errs.append(math.degrees(Rotation.from_matrix(T_err[:3, :3]).magnitude()))
        return float(np.mean(angle_errs))

    def _optimize_component(self, g, edges, comp, ref, poses):
        others = [c for c in comp if c != ref]
        idx = {c: i for i, c in enumerate(others)}

        def unpack(x):
            result = {ref: np.eye(4)}
            for c in others:
                rv, t = x[6 * idx[c]:6 * idx[c] + 3], x[6 * idx[c] + 3:6 * idx[c] + 6]
                T = np.eye(4)
                T[:3, :3] = Rotation.from_rotvec(rv).as_matrix()
                T[:3, 3] = t
                result[c] = T
            return result

        x0 = np.zeros(6 * len(others))
        for c in others:
            R, t = T_to_rt(poses[c])
            x0[6 * idx[c]:6 * idx[c] + 3] = Rotation.from_matrix(R).as_rotvec()
            x0[6 * idx[c] + 3:6 * idx[c] + 6] = t

        edge_list = list(g.subgraph(comp).edges())

        def residuals(x):
            cur = unpack(x)
            res = []
            for a, b in edge_list:
                T_meas = self._edge_transform(edges, a, b)
                T_pred = cur[b] @ invert_T(cur[a])
                T_err = invert_T(T_meas) @ T_pred
                weight = math.sqrt(edges[tuple(sorted((a, b)))]["count"])
                res.extend((Rotation.from_matrix(T_err[:3, :3]).as_rotvec() * weight).tolist())
                res.extend((T_err[:3, 3] * weight).tolist())
            return np.array(res)

        result = least_squares(residuals, x0, method="lm")
        return unpack(result.x)

    def solve(self, cam_ids):
        edges = self.aggregate_edges()
        g = nx.Graph()
        g.add_nodes_from(cam_ids)
        for (a, b), e in edges.items():
            g.add_edge(a, b, weight=e["mean_reproj_error_px"], **e)

        result = {
            "poses": {}, "components": [], "reference_cameras": {},
            "excluded": [], "chain_consistency_error_deg": {}, "method": {},
        }
        ec = self.cfg["extrinsics"]
        for comp in nx.connected_components(g):
            comp = sorted(comp)
            if len(comp) == 1:
                result["excluded"].append(comp[0])
                continue

            ref = max(
                comp,
                key=lambda c: (g.degree(c), sum(g[c][n]["count"] for n in g.neighbors(c))),
            )
            poses = {ref: np.eye(4)}
            for cam in comp:
                if cam != ref:
                    poses[cam] = self._chain_pose(g, edges, ref, cam)

            chain_err = self._chain_consistency_error_deg(g, edges, poses, comp)
            method = "naive_chain"
            if ec["use_pose_graph_optimization"] and chain_err > ec["naive_chain_error_threshold_deg"]:
                poses = self._optimize_component(g, edges, comp, ref, poses)
                method = "graph_optimization"

            for cam in comp:
                result["poses"][cam] = poses[cam]
            result["reference_cameras"][ref] = comp
            result["components"].append(comp)
            result["chain_consistency_error_deg"][ref] = chain_err
            result["method"][ref] = method
        return result


def draw_coverage_overlay(frame, coverage):
    """Tints covered grid cells green and uncovered ones red directly on the
    live thumbnail -- a best-effort coverage-guidance heuristic (spec section
    8), not a precisely optimal "go here next" suggestion.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    rows, cols = coverage.rows, coverage.cols
    cell_h, cell_w = h / rows, w / cols
    overlay = out.copy()
    for r in range(rows):
        for c in range(cols):
            x0, y0 = int(c * cell_w), int(r * cell_h)
            x1, y1 = int((c + 1) * cell_w), int((r + 1) * cell_h)
            color = (0, 140, 0) if coverage.filled[r, c] else (0, 0, 200)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
    cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)
    for r in range(1, rows):
        y = int(r * cell_h)
        cv2.line(out, (0, y), (w, y), (90, 90, 90), 1)
    for c in range(1, cols):
        x = int(c * cell_w)
        cv2.line(out, (x, 0), (x, h), (90, 90, 90), 1)
    return out


class ViserManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.server = viser.ViserServer(port=cfg["viser"]["port"])
        self.server.gui.configure_theme(dark_mode=True)
        # viser has no plain background-color setter; a solid-color background image
        # is the documented way to get one.
        self.server.scene.set_background_image(np.zeros((2, 2, 3), dtype=np.uint8))
        self.cam_images = {}
        self.cam_status_md = {}
        self.frustums = {}
        self.board_frames = {}
        self.board_meshes = {}
        self.unknown_labels = {}

    def add_camera_settings_panel(self):
        """Static (non-updating) readout of this run's fixed camera settings."""
        cam_cfg = self.cfg["camera"]
        with self.server.gui.add_folder("Camera Settings"):
            self.server.gui.add_markdown(
                f"**fps**: {cam_cfg['fps']}\n\n"
                f"**shutter**: {cam_cfg['shutter_us']} us\n\n"
                f"**iso**: {cam_cfg['iso']}\n\n"
                f"**resolution**: {cam_cfg['record_width']}x{cam_cfg['record_height']}"
            )

    def add_world_alignment_button(self, on_click):
        """`on_click` is called with no arguments the instant the button is pressed;
        it should just flag a request (the actual capture happens on the next tick
        that has a fresh board detection, since the board pose isn't known here).
        """
        with self.server.gui.add_folder("World Alignment"):
            self.server.gui.add_markdown(
                "Place the board flat on the floor where a camera can see it, then "
                "click below. The world's up/down and horizontal axes will be set "
                "from the board's orientation."
            )
            button = self.server.gui.add_button("Set down direction from board")
            button.on_click(lambda _: on_click())
        return button

    def add_global_controls(self, cmd_queue):
        """Convenience buttons (spec section 8). These only enqueue the same
        commands the stdin CLI accepts -- the CLI/main loop remains the sole
        place that actually acts on them, per spec's "source of truth for
        control flow is the CLI process, not viser callbacks" guidance.
        """
        with self.server.gui.add_folder("Controls"):
            save_button = self.server.gui.add_button("Save now")
            save_button.on_click(lambda _: cmd_queue.put("save"))
            reset_button = self.server.gui.add_button("Reset all")
            reset_button.on_click(lambda _: cmd_queue.put("reset all"))

    def _ensure_camera_panel(self, cam_id):
        if cam_id in self.cam_images:
            return
        with self.server.gui.add_folder(cam_id):
            self.cam_images[cam_id] = self.server.gui.add_image(
                np.zeros((4, 4, 3), dtype=np.uint8), label="live"
            )
            self.cam_status_md[cam_id] = self.server.gui.add_markdown("_waiting for data..._")

    def update_thumbnail(self, cam_id, frame_bgr, coverage):
        self._ensure_camera_panel(cam_id)
        overlaid = draw_coverage_overlay(frame_bgr, coverage)
        self.cam_images[cam_id].image = cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB)

    def update_thumbnail_raw(self, cam_id, frame_bgr):
        """Like update_thumbnail, without the coverage-grid overlay -- for callers
        that have no CameraCalibState (e.g. hand_capture_live.py's tracking mode,
        where calibration state is no longer tracked once poses are frozen).
        """
        self._ensure_camera_panel(cam_id)
        self.cam_images[cam_id].image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def update_status(self, cam_id, state, extra_lines=()):
        self._ensure_camera_panel(cam_id)
        conn = "connected" if state.connected else "**DISCONNECTED**"
        reproj = f"{state.reproj_error:.3f} px" if state.has_intrinsics_estimate else "n/a"
        lines = [
            f"**{cam_id}** -- {conn}",
            f"samples: {state.sample_count()}",
            f"reproj error: {reproj}",
            f"coverage: {state.coverage.ratio() * 100:.0f}%",
            f"converged: {'yes' if state.converged else 'no'}",
            *extra_lines,
        ]
        self.cam_status_md[cam_id].content = "\n\n".join(lines)

    def update_camera_pose(self, cam_id, T_cam_to_world, K, width, height):
        fov = 2 * math.atan(height / (2 * K[1, 1]))
        aspect = width / height
        wxyz = R_to_wxyz(T_cam_to_world[:3, :3])
        position = T_cam_to_world[:3, 3]
        if cam_id in self.frustums:
            handle = self.frustums[cam_id]
            handle.wxyz, handle.position = wxyz, position
        else:
            self.frustums[cam_id] = self.server.scene.add_camera_frustum(
                f"/cameras/{cam_id}", fov=fov, aspect=aspect, scale=0.12,
                wxyz=wxyz, position=position, color=(60, 140, 220),
            )
        label = self.unknown_labels.pop(cam_id, None)
        if label is not None:
            label.remove()

    def mark_pose_unknown(self, cam_id):
        if cam_id in self.frustums or cam_id in self.unknown_labels:
            return
        self.unknown_labels[cam_id] = self.server.scene.add_label(
            f"/cameras_unknown/{cam_id}", f"{cam_id}: pose unknown", position=(0.0, 0.0, 0.0),
        )

    def _board_footprint(self):
        b = self.cfg["board"]
        w = b["squares_x"] * b["square_size_m"]
        h = b["squares_y"] * b["square_size_m"]
        vertices = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        return vertices, faces

    def update_board_pose(self, cam_id, T_board_to_world):
        """Shows the board's actual physical footprint (not just an axis triad) at
        the pose most recently observed by `cam_id`, so the operator can see where
        the board is right now rather than just an abstract coordinate frame.
        """
        wxyz = R_to_wxyz(T_board_to_world[:3, :3])
        position = T_board_to_world[:3, 3]
        if cam_id in self.board_frames:
            handle = self.board_frames[cam_id]
            handle.wxyz, handle.position, handle.visible = wxyz, position, True
            mesh = self.board_meshes[cam_id]
            mesh.wxyz, mesh.position, mesh.visible = wxyz, position, True
        else:
            self.board_frames[cam_id] = self.server.scene.add_frame(
                f"/board/{cam_id}/axis", axes_length=0.08, axes_radius=0.004,
                wxyz=wxyz, position=position,
            )
            vertices, faces = self._board_footprint()
            self.board_meshes[cam_id] = self.server.scene.add_mesh_simple(
                f"/board/{cam_id}/footprint", vertices, faces,
                color=(230, 210, 60), opacity=0.35, side="double",
                wxyz=wxyz, position=position,
            )

    def hide_board_pose(self, cam_id):
        handle = self.board_frames.get(cam_id)
        if handle is not None:
            handle.visible = False
        mesh = self.board_meshes.get(cam_id)
        if mesh is not None:
            mesh.visible = False


# ==============================================================================
# Output
# ==============================================================================
def build_output(cam_ids, calib_states, pose_result, cfg, device_ids=None):
    device_ids = device_ids or {}
    pose_result = pose_result or {
        "poses": {}, "reference_cameras": {}, "excluded": list(cam_ids),
        "components": [], "chain_consistency_error_deg": {}, "method": {},
    }
    ref_of = {cam: ref for ref, comp in pose_result["reference_cameras"].items() for cam in comp}

    cameras_out = {}
    for cam_id in cam_ids:
        state = calib_states[cam_id]
        entry = {
            "device_id": device_ids.get(cam_id),
            "intrinsics": {
                "camera_matrix": state.K.tolist(),
                "dist_coeffs": np.asarray(state.dist).ravel().tolist(),
                "image_width": state.image_size[0],
                "image_height": state.image_size[1],
                "reprojection_error_px": state.reproj_error if state.has_intrinsics_estimate else None,
                "coverage_ratio": state.coverage.ratio(),
                "num_samples": state.sample_count(),
                "converged": state.converged,
            },
            "extrinsics": None,
        }
        if cam_id in pose_result["poses"]:
            R, t = T_to_rt(pose_result["poses"][cam_id])  # reference-camera-frame -> this camera
            ref = ref_of.get(cam_id)
            entry["extrinsics"] = {
                "rotation": R.tolist(),
                "translation": t.tolist(),
                "reference_camera": ref,
                "is_reference": cam_id == ref,
                "chain_consistency_error_deg": pose_result["chain_consistency_error_deg"].get(ref),
                "method": pose_result["method"].get(ref),
            }
        cameras_out[cam_id] = entry

    return {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "board": cfg["board"],
            "reference_camera_selection": (
                "auto-selected as the most-connected node per pose-graph component (ties "
                "broken by total edge observation count) -- an open assumption not specified "
                "by the user, see spec section 12 / this file's module docstring"
            ),
            "reference_cameras": pose_result["reference_cameras"],
            "excluded_cameras": {
                "cameras": pose_result["excluded"],
                "reason": "never shared a board view with any other connected camera",
            },
            "components": pose_result["components"],
        },
        "cameras": cameras_out,
    }


def calibration_output_path(cam_ids, cfg):
    """Each save gets its own timestamped file (never overwrites a previous save),
    named so a picker can list them without opening the file: date, time, camera count.
    """
    out_dir = cfg["output"]["dir"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"{ts}_{len(cam_ids)}cam.json")


def save_output(cam_ids, calib_states, pose_result, cfg, device_ids=None):
    path = calibration_output_path(cam_ids, cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(build_output(cam_ids, calib_states, pose_result, cfg, device_ids), f, indent=2)
    print(f"[Save] Wrote {path}")
    return path


def load_calibration_output(path):
    """Inverse of build_output -- the bridge a hand-tracking app loads a saved
    calibration through. Cameras that were never posed (extrinsics is null) are
    skipped entirely, matching how build_output/PoseGraph already treat "excluded"
    cameras.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    cameras = {}
    for cam_id, entry in payload["cameras"].items():
        if entry["extrinsics"] is None:
            continue
        intr = entry["intrinsics"]
        R = np.array(entry["extrinsics"]["rotation"], dtype=np.float64)
        t = np.array(entry["extrinsics"]["translation"], dtype=np.float64)
        cameras[cam_id] = {
            "K": np.array(intr["camera_matrix"], dtype=np.float64),
            "dist": np.array(intr["dist_coeffs"], dtype=np.float64),
            "width": intr["image_width"],
            "height": intr["image_height"],
            "R": R,
            "t": t,
            "device_id": entry.get("device_id"),
        }
    return cameras


# ==============================================================================
# CLI / main loop
# ==============================================================================
def read_stdin_commands(cmd_queue, stop_event):
    while not stop_event.is_set():
        try:
            line = input()
        except EOFError:
            break
        cmd_queue.put(line.strip().lower())


def try_reconnect(session, cam_id):
    print(f"[Reconnect] Attempting to reconnect {cam_id} ({session.device_id})...")
    try:
        available = dai.Device.getAllAvailableDevices()
        match = next((d for d in available if d.deviceId == session.device_id), None)
        if match is None:
            print(f"[Reconnect] {cam_id} not found among available devices yet.")
            return
        session.device_info = match
        session._connect()
        print(f"[Reconnect] {cam_id} reconnected.")
    except Exception as exc:
        print(f"[Reconnect] {cam_id} reconnect failed: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Multi-camera live ChArUco calibration")
    parser.add_argument("--config", default=CONFIG_PATH_DEFAULT, help="Path to the YAML config file")
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Optional auto-stop after N seconds (mainly for headless/automated runs)",
    )
    args = parser.parse_args()

    if dai is None:
        print("[Error] depthai is not installed in this environment.")
        return

    cfg = load_config(args.config)
    _board, detector, board_points_3d = build_board(cfg)

    device_infos = dai.Device.getAllAvailableDevices()
    if not device_infos:
        print("[Error] No OAK devices discovered.")
        return
    print(f"Found {len(device_infos)} OAK device(s).")

    cam_ids = [f"cam{i}" for i in range(len(device_infos))]
    sessions, calib_states = {}, {}
    image_size = (cfg["camera"]["record_width"], cfg["camera"]["record_height"])
    try:
        for cam_id, info in zip(cam_ids, device_infos):
            print(f"Starting {cam_id} ({info.deviceId})...")
            sessions[cam_id] = CalibCameraSession(cam_id, info, cfg)
            calib_states[cam_id] = CameraCalibState(cam_id, image_size, cfg)
    except Exception as exc:
        print(f"[Error] Failed to start cameras: {exc}")
        for session in sessions.values():
            session.close()
        return

    if len(cam_ids) < 2:
        print("[Warning] Only one camera detected -- extrinsics are meaningless with a single "
              "camera. Intrinsic calibration still runs and will be reported.")

    pose_graph = PoseGraph(cfg)
    viser_mgr = ViserManager(cfg)
    viser_mgr.add_camera_settings_panel()
    print(f"[Viser] http://localhost:{viser_mgr.server.get_port()}")
    print("Type 'save' to write results now, 'reset <camN|all>' to clear samples, 'quit' to stop.\n")

    cmd_queue = queue.Queue()
    stop_event = threading.Event()
    threading.Thread(target=read_stdin_commands, args=(cmd_queue, stop_event), daemon=True).start()
    viser_mgr.add_global_controls(cmd_queue)

    world_align = {"pending": False, "R": None}

    def request_world_alignment():
        world_align["pending"] = True
        print("[Align] Waiting for a fresh board detection to set the world's down direction...")

    viser_mgr.add_world_alignment_button(request_world_alignment)

    runtime_cfg = cfg["runtime"]
    tol_s = cfg["extrinsics"]["simultaneity_tolerance_ms"] / 1000.0
    last_status_print = time.monotonic()
    last_reconnect_attempt = {cam_id: 0.0 for cam_id in cam_ids}
    start_time = time.monotonic()
    pose_result = None

    try:
        while True:
            if args.duration is not None and time.monotonic() - start_time > args.duration:
                print(f"[Duration] {args.duration}s elapsed -- stopping.")
                break

            while not cmd_queue.empty():
                cmd = cmd_queue.get()
                if cmd in ("q", "quit", "exit"):
                    raise KeyboardInterrupt
                if cmd == "save":
                    device_ids = {cam_id: sessions[cam_id].device_id for cam_id in cam_ids}
                    save_output(cam_ids, calib_states, pose_result, cfg, device_ids)
                elif cmd.startswith("reset"):
                    parts = cmd.split()
                    target = parts[1] if len(parts) > 1 else "all"
                    for cam_id in (cam_ids if target == "all" else [target]):
                        if cam_id in calib_states:
                            calib_states[cam_id] = CameraCalibState(cam_id, image_size, cfg)
                            print(f"[Reset] {cam_id} calibration state cleared.")
                else:
                    print(f"[Command] Unrecognized: {cmd!r}")

            now = time.monotonic()

            # Pass 1: fetch/decode every camera's latest frame first, as fast as possible,
            # and timestamp it right there. Detection + PnP + occasional bundle-adjustment
            # recalibration below are comparatively slow (tens to hundreds of ms per camera);
            # if the "simultaneous" timestamp were taken after that work instead, cameras
            # processed later in the same tick would look falsely delayed relative to
            # cameras processed first, and few/no frame pairs would ever fall within the
            # simultaneity tolerance -- which otherwise silently starves the pose graph of
            # edges even though the cameras really did see the board at the same time.
            fresh_frames = {}
            for cam_id in cam_ids:
                session, state = sessions[cam_id], calib_states[cam_id]

                if not session.connected:
                    if now - last_reconnect_attempt[cam_id] >= runtime_cfg["reconnect_retry_interval_s"]:
                        last_reconnect_attempt[cam_id] = now
                        try_reconnect(session, cam_id)
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

            # Pass 2: the slow per-camera detection/calibration work, using the frame-arrival
            # timestamp captured above rather than a fresh one taken after this work runs.
            for cam_id, (frame, frame_ts) in fresh_frames.items():
                state = calib_states[cam_id]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detection = detect_charuco(detector, gray)
                if detection is None:
                    # No board seen this tick -- hide it rather than leaving this
                    # camera's last-ever sighting stuck on screen forever.
                    viser_mgr.hide_board_pose(cam_id)
                    continue
                corners2d, ids = detection

                pose, accepted = process_detection(state, corners2d, ids, board_points_3d, gray, cfg)
                if accepted:
                    maybe_recalibrate(state, cfg)
                if pose is None:
                    viser_mgr.hide_board_pose(cam_id)
                    continue
                R, t, err = pose
                state.last_detection_ts = frame_ts
                state.last_edge_pose = (frame_ts, R, t, err)
                state.last_warned_no_detection = False

                if pose_result is not None and cam_id in pose_result["poses"]:
                    T_cam_to_world = invert_T(pose_result["poses"][cam_id])
                    viser_mgr.update_board_pose(cam_id, T_cam_to_world @ rt_to_T(R, t))
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
                # Always resolve (even with zero edges so far) so every camera gets at least
                # a "pose unknown" placeholder in the scene from the first tick onward, instead
                # of the 3D view showing nothing at all until the first edge is observed.
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
                        T_cam_to_world = invert_T(pose_result["poses"][cam_id])
                        world_align["R"] = compute_board_world_alignment(T_cam_to_world, R_bc, t_bc)
                        world_align["pending"] = False
                        print(f"[Align] World down direction set from {cam_id}'s board detection.")
                        break
                apply_world_alignment(pose_result, world_align["R"])

                for cam_id in cam_ids:
                    state = calib_states[cam_id]
                    if cam_id in pose_result["poses"]:
                        T_cam_to_world = invert_T(pose_result["poses"][cam_id])
                        viser_mgr.update_camera_pose(cam_id, T_cam_to_world, state.K, *state.image_size)
                    else:
                        viser_mgr.mark_pose_unknown(cam_id)

            for cam_id in cam_ids:
                state = calib_states[cam_id]
                extra = []
                if state.last_detection_ts is not None:
                    silent_for = time.time() - state.last_detection_ts
                    if silent_for > runtime_cfg["no_detection_warning_s"]:
                        extra.append(f"**no detection for {silent_for:.0f}s**")
                        if not state.last_warned_no_detection:
                            print(f"[Warning] {cam_id}: no board detection for {silent_for:.0f}s")
                            state.last_warned_no_detection = True
                elif state.connected:
                    extra.append("**never detected the board yet**")
                viser_mgr.update_status(cam_id, state, extra)

            if now - last_status_print >= runtime_cfg["status_print_interval_s"]:
                last_status_print = now
                parts = [
                    f"{c}: {calib_states[c].sample_count()} samples, "
                    f"reproj={calib_states[c].reproj_error:.2f}px, "
                    f"cov={calib_states[c].coverage.ratio() * 100:.0f}%, "
                    f"{'OK' if calib_states[c].converged else '...'}"
                    for c in cam_ids
                ]
                print("[Status] " + " | ".join(parts))

            if (
                runtime_cfg["auto_stop"] and cam_ids
                and all(calib_states[c].converged for c in cam_ids)
                and (len(cam_ids) < 2 or pose_result is not None)
            ):
                print("[Auto-stop] All cameras converged.")
                break

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n[Stop] Interrupted by user.")

    finally:
        stop_event.set()
        device_ids = {cam_id: sessions[cam_id].device_id for cam_id in cam_ids}
        for session in sessions.values():
            session.close()
        save_output(cam_ids, calib_states, pose_result, cfg, device_ids)


if __name__ == "__main__":
    main()
