"""fast3d: CPU-only 3D MediaPipe skeleton (body, hands) and face for one take.

    python -m fast3d.run <take_dir> [--calib PATH] [--workers 8]

Steps (each is a module of this package):
  1. frame sync: align_session.plan_alignment (same timestamp matching as
     align_session, no JPEGs written);
  2. MediaPipe 2D per camera, cameras in parallel threads
     (fast3d.extract2d); --reuse-2d loads the saved 2D instead;
  3. calibration (fast3d.calib): --calib, else the one linked to the take,
     else take_meta's, else the best-scoring matching calibration; scored
     with fast3d.calib_check on the take's 2D (verdict ok / warning / bad,
     flagged cameras); without one only the 2D step runs;
  4. triangulation per part (fast3d.triangulate).

Outputs in <take_dir>/processed/fast3d/:
  fast3d.npz      frame_keys, for each part P in body, left, right, face:
                  P_raw / P_smoothed (T, N, 3) metres in the calibration frame
                  (NaN where not triangulated), P_ncams (T,); edges_body,
                  edges_hand, edges_face (FaceMesh tessellation) index pairs; tris_face (898, 3)
                  triangles of MediaPipe's canonical face model (points 0-467);
                  cam_R, cam_t, cam_K, cams, image_size (for drawing the cameras), fps
  fast3d_2d.npz   the per-camera 2D detections (raw image, normalised)
  fast3d_meta.json  calibration used and why, its check and the scored
                  candidates, times per step, frames found
                  per part and camera, triangulated frames and median
                  reprojection error per part
and <take_dir>/processed/fast3d_status.json: status (processing / done /
failed / done_2d_only), stage, progress 0-1, error, calibration (path,
how, verdict, median_px, flagged cameras); written atomically,
read by the app.
"""
import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def status_path(take_dir):
    return os.path.join(take_dir, "processed", "fast3d_status.json")


def write_status(take_dir, **fields):
    os.makedirs(os.path.join(take_dir, "processed"), exist_ok=True)
    p = status_path(take_dir)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(fields, updated_unix=time.time()), f)
    os.replace(tmp, p)


def read_status(take_dir):
    try:
        with open(status_path(take_dir), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def edges():
    import mediapipe as mp
    s = mp.solutions
    return (np.array(sorted(s.pose.POSE_CONNECTIONS), np.int32), np.array(sorted(s.hands.HAND_CONNECTIONS), np.int32),
            np.array(sorted(s.face_mesh.FACEMESH_TESSELATION), np.int32))


VERDICT_RANK = {"ok": 0, "warning": 1, "bad": 2}
# One process: the cameras' 2D runs in threads (MediaPipe releases the GIL; 8 cameras take about 1.6x one camera),
# the triangulation runs part after part. An earlier version used process pools; on signbot1 (i9-14900K) about 7 %
# of processes importing matplotlib in parallel crashed with corrupted-interpreter errors that never happen
# sequentially, the known instability of that CPU generation, and loading the libraries once is cheaper anyway.


def run(take_dir, calib_path=None, workers=8, reuse_2d=False, calib_dirs=None):
    import align_session
    import calibrate
    from fast3d import calib_check
    from fast3d.calib import DEFAULT_SEARCH_DIRS, candidate_calibrations, choose_calibration, read_link
    from fast3d.triangulate import PARTS, triangulate, undistort_norm
    from pose2d.triangulation import build_projection_matrices

    take_dir = os.path.abspath(take_dir)
    out = os.path.join(take_dir, "processed", "fast3d")
    os.makedirs(out, exist_ok=True)
    times, meta = {}, {}
    write_status(take_dir, status="processing", stage="sync", progress=0.0)

    t0 = time.time()
    cams, idx, info = align_session.plan_alignment(take_dir)
    T = len(idx[cams[0]])
    keys = [f"{i:06d}.jpg" for i in range(T)]
    w, h = info["rec_w"], info["rec_h"]
    vp = {c["label"]: c["video_path"] for c in info["cameras"]}
    times["sync"] = time.time() - t0

    t0 = time.time()
    p2d = os.path.join(out, "fast3d_2d.npz")
    if reuse_2d and os.path.isfile(p2d):
        z = np.load(p2d)
        names = ("body", "body_vis", "left", "left_score", "right", "right_score", "face", "face_found")
        per_cam = {c: {k: z[f"{c}_{k}"] for k in names} for c in cams}
        times["2d_reused"] = time.time() - t0
    else:
        write_status(take_dir, status="processing", stage="2d", progress=0.0)
        from fast3d.extract2d import extract_camera
        per_cam = {}
        with ThreadPoolExecutor(max_workers=min(workers, len(cams))) as ex:
            futs = {ex.submit(extract_camera, vp[c], idx[c], w, h): c for c in cams}
            for f in as_completed(futs):
                per_cam[futs[f]] = f.result()
                write_status(take_dir, status="processing", stage="2d", progress=len(per_cam) / len(cams))
        per_cam = {c: per_cam[c] for c in cams}
        times["2d"] = time.time() - t0
        np.savez_compressed(p2d, frame_keys=np.array(keys), cams=np.array(cams),
                            **{f"{c}_{k}": v for c, d in per_cam.items() for k, v in d.items()})
    meta["frames_found"] = {c: {"body": int(np.isfinite(d["body"][:, 0, 0]).sum()), "left": int((d["left_score"] > 0).sum()),
                                "right": int((d["right_score"] > 0).sum()), "face": int(d["face_found"].sum())}
                            for c, d in per_cam.items()}

    # calibration: given > linked by hand > take_meta > best-scoring matching candidate
    t0 = time.time()
    write_status(take_dir, status="processing", stage="calibration", progress=0.0)
    body = {c: d["body"] for c, d in per_cam.items()}
    vis = {c: d["body_vis"] for c, d in per_cam.items()}

    def score(path):
        cal = calibrate.load_calibration_output(path)
        if any(c not in cal for c in cams):
            return cal, {"verdict": "bad", "error": "cameras missing", "median_px": float("inf")}
        return cal, calib_check.check(body, vis, cams, cal, w, h)

    search_dirs = tuple(calib_dirs or ()) + tuple(d for d in DEFAULT_SEARCH_DIRS if d not in (calib_dirs or ()))
    how = "given" if calib_path else None
    if calib_path is None and read_link(take_dir):
        calib_path, how = read_link(take_dir), "linked"
    if calib_path is None:
        p, h_ = choose_calibration(take_dir, search_dirs=search_dirs, repo_root=REPO)
        if h_ == "take_meta":
            calib_path, how = p, "take_meta"
    if calib_path is not None:
        calib, chk = score(calib_path)
        meta["calibration_candidates"] = []
    else:
        cands = []
        for p, dt in candidate_calibrations(take_dir, search_dirs=search_dirs, repo_root=REPO):
            cal, chk = score(p)
            cands.append((VERDICT_RANK[chk["verdict"]], chk["median_px"], dt, p, cal, chk))
        meta["calibration_candidates"] = [{"path": c[3], "verdict": c[5]["verdict"], "median_px": c[1], "dt_s": c[2]}
                                          for c in sorted(cands, key=lambda c: c[:3])]
        if cands:
            _, _, dt, calib_path, calib, chk = min(cands, key=lambda c: c[:3])
            how = f"best of {len(cands)} matching ({dt:.0f} s from take start)"
        else:
            calib, chk, how = None, None, "none"
    meta["calibration"] = {"path": calib_path, "how": how}
    if chk is not None:
        meta["calibration"]["check"] = chk
    times["calibration"] = time.time() - t0
    calib_status = None if chk is None else {"path": calib_path, "how": how, "verdict": chk["verdict"],
                                             "median_px": chk.get("median_px"), "flagged": chk.get("flagged", [])}

    eb, eh, ef = edges()
    res = dict(frame_keys=np.array(keys), edges_body=eb, edges_hand=eh, edges_face=ef, cams=np.array(cams),
               image_size=np.array([w, h]), fps=info["fps"],
               tris_face=np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mediapipe_face_triangles.npy")))
    if calib is None:
        meta["times_s"] = times
        _write_meta(out, meta)
        np.savez_compressed(os.path.join(out, "fast3d.npz"), **res)
        write_status(take_dir, status="done_2d_only", stage="done", progress=1.0,
                     error="no matching calibration; 2D only", calibration=None)
        return meta

    t0 = time.time()
    write_status(take_dir, status="processing", stage="triangulate", progress=0.0)
    und = {}
    for c, d in per_cam.items():
        K, dist = calib[c]["K"], calib[c]["dist"]
        und[c] = {k: (undistort_norm(v, K, dist, w, h) if k in ("body", "left", "right", "face") else v) for k, v in d.items()}
    P = build_projection_matrices(calib, cams)
    meta["triangulation"] = {}
    for n, p in enumerate(PARTS, 1):
        r = triangulate(p, und, keys, P, w, h)
        res[f"{p}_raw"], res[f"{p}_smoothed"], res[f"{p}_ncams"] = r["raw"], r["smoothed"], r["n_cams"]
        meta["triangulation"][p] = {"frames": int(np.isfinite(r["raw"][:, PARTS[p][1], 0]).sum()),
                                    "median_reproj_px": r["reproj_px"]}
        write_status(take_dir, status="processing", stage="triangulate", progress=n / len(PARTS))
    times["triangulate"] = time.time() - t0
    res.update(cam_R=np.stack([calib[c]["R"] for c in cams]), cam_t=np.stack([np.ravel(calib[c]["t"]) for c in cams]),
               cam_K=np.stack([calib[c]["K"] for c in cams]))
    np.savez_compressed(os.path.join(out, "fast3d.npz"), **res)
    meta["times_s"] = times
    meta["n_frames"], meta["fps"], meta["cams"] = T, info["fps"], cams
    _write_meta(out, meta)
    write_status(take_dir, status="done", stage="done", progress=1.0, calibration=calib_status)
    return meta


def _write_meta(out, meta):
    with open(os.path.join(out, "fast3d_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take_dir")
    ap.add_argument("--calib", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--calib-dir", action="append", default=[], help="extra calibration search dir (repeatable)")
    ap.add_argument("--reuse-2d", action="store_true", help="reuse processed/fast3d/fast3d_2d.npz (e.g. to change calibration)")
    ap.add_argument("--link", action="store_true", help="also link --calib to the take (calibration_link.json)")
    a = ap.parse_args()
    if a.link and a.calib:
        from fast3d.calib import write_link
        write_link(os.path.abspath(a.take_dir), a.calib, set_by="fast3d.run --link")
    t = time.time()
    try:
        meta = run(a.take_dir, a.calib, a.workers, a.reuse_2d, a.calib_dir)
    except Exception as exc:
        write_status(os.path.abspath(a.take_dir), status="failed", stage="error", progress=0.0, error=str(exc),
                     traceback=traceback.format_exc()[-4000:])
        traceback.print_exc()
        sys.exit(1)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("frames_found", "triangulation")}, indent=1))
    print(f"fast3d total {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
