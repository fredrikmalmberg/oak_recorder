import os
import glob
import json
import argparse
import numpy as np
import cv2


def collect_images(folder):
    exts = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]
    files = set()
    for e in exts:
        files.update(glob.glob(os.path.join(folder, e)))
    return sorted(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Folder with images from ONE camera")
    ap.add_argument("--out", default="charuco_calib_out", help="Output folder")

    # ChArUco board parameters
    ap.add_argument("--aruco_dict", default="DICT_4X4_50", help="OpenCV ArUco dict name (e.g., DICT_4X4_50)")
    ap.add_argument("--squares-x", type=int, required=True, help="Number of chessboard squares in X")
    ap.add_argument("--squares-y", type=int, required=True, help="Number of chessboard squares in Y")
    ap.add_argument("--square-size", type=float, required=True, help="Chessboard square size (real units, e.g., mm)")
    ap.add_argument("--marker-size", type=float, required=True, help="Marker size (real units, e.g., mm)")

    # Data selection
    ap.add_argument("--min-corners", type=int, default=10, help="Minimum detected Charuco corners per image")
    ap.add_argument("--min-views", type=int, default=10, help="Minimum valid views for calibration")

    ap.add_argument("--use-intrinsic-guess", action="store_true", help="Use simple intrinsics init guess")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if not os.path.isdir(args.images):
        raise RuntimeError(f"--images is not a folder: {args.images}")

    images = collect_images(args.images)
    if not images:
        raise RuntimeError("No images found in --images")

    # OpenCV 5.0.0: use CharucoDetector (no interpolateCornersCharuco in your build)
    aruco_dict_id = getattr(cv2.aruco, args.aruco_dict, None)
    if aruco_dict_id is None:
        raise ValueError(
            f"Unknown --aruco_dict '{args.aruco_dict}'. "
            "Try one of: " + ", ".join([n for n in dir(cv2.aruco) if n.startswith("DICT_4X4_")])
        )

    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_size,
        args.marker_size,
        aruco_dict,
    )

    detector_params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    charuco_detector = cv2.aruco.CharucoDetector(board, detectorParams=detector_params)

    used_images = []
    objpoints = []  # list of (N,3)
    imgpoints = []  # list of (N,2)

    image_size = None

    # board.chessboardCorners corresponds to chessboard corner coordinates in board frame
    # res.ids are indices into those chessboard corners.
    board_corners_3d = np.asarray(board.getChessboardCorners(), dtype=np.float64)

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping unreadable: {img_path}")
            continue

        if image_size is None:
            h0, w0 = img.shape[:2]
            image_size = (w0, h0)  # (w,h)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)

        if charuco_ids is None:
            continue

        ids = np.array(charuco_ids).reshape(-1)
        if ids.size < args.min_corners:
            continue

        # 2D points
        corners2d = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)

        # 3D points for those ids
        if ids.max(initial=-1) >= board_corners_3d.shape[0]:
            # Skip if something is inconsistent
            continue

        obj = board_corners_3d[ids]  # (N,3)

        if corners2d.shape[0] != ids.shape[0]:
            # Safety check
            continue

        objpoints.append(obj.astype(np.float32).reshape(-1, 1, 3))
        imgpoints.append(corners2d.astype(np.float32).reshape(-1, 1, 2))
        used_images.append(img_path)

        print(f"Used {ids.size} corners from {os.path.basename(img_path)}")

    if len(used_images) < args.min_views:
        raise RuntimeError(
            f"Only {len(used_images)} valid views (need >= {args.min_views}). "
            f"Try better images, more views, or lowering --min-corners."
        )

    w, h = image_size

    if args.use_intrinsic_guess:
        camera_matrix_init = np.array(
            [[w, 0, w / 2.0],
             [0, h, h / 2.0],
             [0, 0, 1.0]],
            dtype=np.float64,
        )
        dist_init = np.zeros((5, 1), dtype=np.float64)
        flags = cv2.CALIB_USE_INTRINSIC_GUESS
        init_desc = "simple center/scale guess"
    else:
        camera_matrix_init = np.zeros((3, 3), dtype=np.float64)
        dist_init = np.zeros((5, 1), dtype=np.float64)
        flags = 0
        init_desc = "no intrinsic guess"

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=objpoints,
        imagePoints=imgpoints,
        imageSize=(w, h),
        cameraMatrix=camera_matrix_init,
        distCoeffs=dist_init,
        flags=flags,
    )

    out_path = os.path.join(args.out, "calibration_result.json")
    result = {
        "image_size": {"w": int(w), "h": int(h)},
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.flatten().tolist(),
        "rmse": float(ret),
        "num_views_used": len(used_images),
        "init_mode": init_desc,
        "used_images": used_images,
        "board": {
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "square_size": args.square_size,
            "marker_size": args.marker_size,
            "aruco_dict": args.aruco_dict,
        },
        "opencv_version": cv2.__version__,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== Calibration done ===")
    print(f"Views used: {len(used_images)}")
    print(f"RMSE: {ret}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
