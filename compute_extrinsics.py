import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def collect_images(folder):
    exts = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]
    files = set()
    for e in exts:
        files.update(glob.glob(os.path.join(folder, e)))
    return sorted(files)


def frame_key(img_path):
    return os.path.splitext(os.path.basename(img_path))[0]


def load_calib(path):
    with open(path) as f:
        d = json.load(f)
    K = np.array(d["camera_matrix"], dtype=np.float64)
    dist = np.array(d["dist_coeffs"], dtype=np.float64)
    return K, dist, d["board"]


def make_charuco_detector(board_cfg):
    dict_id = getattr(cv2.aruco, board_cfg["aruco_dict"])
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard(
        (board_cfg["squares_x"], board_cfg["squares_y"]),
        board_cfg["square_size"],
        board_cfg["marker_size"],
        aruco_dict,
    )
    params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.CharucoDetector(board, detectorParams=params)
    board_corners_3d = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    return detector, board_corners_3d


def detect_board_pose(detector, board_corners_3d, K, dist, img_path, min_corners):
    img = cv2.imread(img_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_ids is None:
        return None

    ids = np.array(charuco_ids).reshape(-1)
    if ids.size < min_corners or ids.max(initial=-1) >= board_corners_3d.shape[0]:
        return None

    obj = board_corners_3d[ids]
    img_pts = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
    if img_pts.shape[0] != ids.shape[0]:
        return None

    ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, dist)
    if not ok:
        return None

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3), int(ids.size)


def average_rotations(rotations, weights):
    """Weighted quaternion average (Markley et al.)."""
    quats = np.array([Rotation.from_matrix(R).as_quat() for R in rotations])  # (N,4) xyzw
    ref = quats[0]
    signs = np.sign(quats @ ref)
    signs[signs == 0] = 1.0
    quats = quats * signs[:, None]

    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()

    M = (quats * w[:, None]).T @ quats
    eigvals, eigvecs = np.linalg.eigh(M)
    q_avg = eigvecs[:, np.argmax(eigvals)]
    return Rotation.from_quat(q_avg).as_matrix()


def parse_calib_args(entries):
    calibs = {}
    for entry in entries:
        cam_id, path = entry.split(":", 1)
        calibs[cam_id] = load_calib(path)
    return calibs


def solve_extrinsics(session, calibs, ref_cam, min_corners):
    poses = {}
    for cam_id, (K, dist, board_cfg) in calibs.items():
        detector, board_corners_3d = make_charuco_detector(board_cfg)
        img_dir = os.path.join(session, "aligned", cam_id)
        images = collect_images(img_dir)
        if not images:
            raise RuntimeError(f"No images found for {cam_id} in {img_dir}")

        cam_poses = {}
        for img_path in images:
            result = detect_board_pose(detector, board_corners_3d, K, dist, img_path, min_corners)
            if result is None:
                continue
            R, t, n = result
            cam_poses[frame_key(img_path)] = (R, t, n)
        print(f"{cam_id}: board detected in {len(cam_poses)}/{len(images)} frames")
        poses[cam_id] = cam_poses

    if ref_cam not in poses:
        raise ValueError(f"--ref-cam {ref_cam} not found among --calib entries")

    ref_poses = poses[ref_cam]
    extrinsics = {ref_cam: {"R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0], "num_frames": len(ref_poses)}}

    for cam_id in calibs:
        if cam_id == ref_cam:
            continue

        cam_poses = poses[cam_id]
        common = sorted(set(ref_poses) & set(cam_poses))
        if not common:
            print(f"WARNING: no common frames between {ref_cam} and {cam_id}, skipping")
            continue

        Rs, ts, weights = [], [], []
        for fk in common:
            R0, t0, n0 = ref_poses[fk]
            Rj, tj, nj = cam_poses[fk]
            R_j0 = Rj @ R0.T
            t_j0 = tj - R_j0 @ t0
            Rs.append(R_j0)
            ts.append(t_j0)
            weights.append(min(n0, nj))

        R_avg = average_rotations(Rs, weights)
        ts_arr = np.array(ts)
        w_arr = np.asarray(weights, dtype=np.float64)
        t_avg = np.average(ts_arr, axis=0, weights=w_arr)

        rot_errs_deg = [
            np.degrees(np.arccos(np.clip((np.trace(R_avg.T @ R) - 1) / 2, -1, 1))) for R in Rs
        ]
        print(
            f"{cam_id} <- {ref_cam}: {len(common)} common frames, "
            f"rot err mean/std {np.mean(rot_errs_deg):.3f}/{np.std(rot_errs_deg):.3f} deg, "
            f"t std {ts_arr.std(axis=0).round(2)} (board units)"
        )

        extrinsics[cam_id] = {
            "R": R_avg.tolist(),
            "t": t_avg.tolist(),
            "num_frames": len(common),
        }

    return extrinsics


def launch_viewer(extrinsics, calibs, session, ref_cam, min_corners, units_per_meter):
    import viser

    server = viser.ViserServer()
    print("Viser viewer running at http://localhost:8080 (Ctrl+C to exit)")

    # Extrinsics/board points are in calibration units (e.g. mm, from --square-size).
    # Viser's default camera framing assumes roughly meter-scale scenes, so rescale
    # everything for display; the saved extrinsics.json keeps the original units.
    s = 1.0 / units_per_meter

    server.scene.add_frame("/world", axes_length=0.2, axes_radius=0.005, show_axes=True)

    cam_centers = []
    colors = [(220, 60, 60), (60, 160, 220), (60, 200, 120), (220, 180, 60)]
    for i, (cam_id, ext) in enumerate(extrinsics.items()):
        R = np.array(ext["R"])
        t = np.array(ext["t"])
        # camera-to-world pose: position is the camera center in the reference frame,
        # orientation rotates points from camera space into that frame.
        cam_center = -R.T @ t
        cam_centers.append(cam_center)
        R_c2w = R.T
        wxyz = Rotation.from_matrix(R_c2w).as_quat()[[3, 0, 1, 2]]  # xyzw -> wxyz

        K, _, _ = calibs[cam_id]
        fov = 2 * np.arctan2(K[1, 2], K[1, 1])
        aspect = K[0, 2] / K[1, 2]

        server.scene.add_camera_frustum(
            f"/world/{cam_id}",
            fov=fov,
            aspect=aspect,
            scale=0.15,
            color=colors[i % len(colors)],
            wxyz=wxyz,
            position=cam_center * s,
        )
        server.scene.add_label(f"/world/{cam_id}/label", cam_id)

    # Overlay the board corners for one shared frame as a sanity check on scale/alignment.
    if ref_cam in calibs:
        _, _, board_cfg = calibs[ref_cam]
        detector, board_corners_3d = make_charuco_detector(board_cfg)
        K0, dist0, _ = calibs[ref_cam]
        img_dir = os.path.join(session, "aligned", ref_cam)
        for img_path in collect_images(img_dir):
            result = detect_board_pose(detector, board_corners_3d, K0, dist0, img_path, min_corners)
            if result is None:
                continue
            R0, t0, _ = result
            board_world = (R0 @ board_corners_3d.T).T + t0
            server.scene.add_point_cloud(
                "/world/board_sample",
                points=board_world * s,
                colors=(255, 255, 255),
                point_size=0.01,
            )
            break

    # Frame the initial view around the cameras instead of relying on viser's
    # meter-scale default, which left a small rig looking tiny and far away.
    centroid = np.mean(cam_centers, axis=0) * s
    radius = max(np.max(np.linalg.norm(np.array(cam_centers) * s - centroid, axis=1)), 0.2)

    @server.on_client_connect
    def _(client):
        client.camera.position = centroid + radius * np.array([1.5, -1.5, 1.2])
        client.camera.look_at = centroid

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


def main():
    ap = argparse.ArgumentParser(description="Compute multi-camera extrinsics from a shared ChArUco board and view them in 3D.")
    ap.add_argument("--session", required=True, help="Recording session folder (contains aligned/camN/*.jpg)")
    ap.add_argument(
        "--calib",
        nargs="+",
        required=True,
        help="cam_id:calibration_result.json pairs, e.g. cam0:camera0_calib/calibration_result.json",
    )
    ap.add_argument("--ref-cam", default="cam0", help="Reference camera; becomes the world origin")
    ap.add_argument("--min-corners", type=int, default=8, help="Minimum ChArUco corners required to accept a frame's pose")
    ap.add_argument("--out", default="extrinsics.json", help="Output JSON path")
    ap.add_argument("--no-viewer", action="store_true", help="Skip launching the viser viewer")
    ap.add_argument(
        "--units-per-meter",
        type=float,
        default=1000.0,
        help="Calibration units per meter, for viewer scaling only (default 1000, i.e. --square-size was in mm)",
    )
    args = ap.parse_args()

    calibs = parse_calib_args(args.calib)
    extrinsics = solve_extrinsics(args.session, calibs, args.ref_cam, args.min_corners)

    with open(args.out, "w") as f:
        json.dump(extrinsics, f, indent=2)
    print(f"Saved extrinsics: {args.out}")

    if not args.no_viewer:
        launch_viewer(extrinsics, calibs, args.session, args.ref_cam, args.min_corners, args.units_per_meter)


if __name__ == "__main__":
    main()
