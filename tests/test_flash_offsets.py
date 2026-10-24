"""Pure-function tests for flash_offsets.py (no video files needed)."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flash_offsets as fo  # noqa: E402

# Candidate frames per camera as detected on take_1 (20261019_180836): flash 1 (RGB) near frame 3, flash 2 (white) near 33-36,
# plus junk from motion. Regression guard for the voting / pairing logic.
TAKE1_EVENTS = {
    "cam0": [(3, 49.3), (12, 34.4), (19, 30.3), (26, 28.4), (36, 40.8)],
    "cam1": [(2, 66.8), (20, 43.3), (32, 31.1), (36, 31.1)],
    "cam2": [(36, 33.9)],
    "cam3": [],
    "cam4": [(3, 47.9), (35, 28.1)],
    "cam5": [(34, 32.4), (21, 47.0), (29, 43.1), (85, 43.9)],
    "cam6": [(7, 36.0), (31, 35.7)],
    "cam7": [(29, 31.7)],
}
CAMS = [f"cam{i}" for i in range(8)]


def test_find_peaks_returns_strongest_frame_of_each_run():
    s = np.zeros(100, np.float32)
    s[10:13] = [30, 80, 40]
    s[60] = 90
    assert fo.find_peaks(s, min_peak=20) == [(11, 80.0), (60, 90.0)]


def test_find_peaks_ignores_noise():
    rng = np.random.default_rng(0)
    assert fo.find_peaks(rng.normal(5, 1, 200).astype(np.float32), min_peak=25) == []


def test_take1_offsets():
    r = fo.compute_offsets(TAKE1_EVENTS, CAMS, 30.0)
    off = {c: r["cameras"][c]["offset_frames"] for c in CAMS}
    assert r["reference"] == "cam0" and r["overall"] == "partial"
    assert off == {"cam0": 0, "cam1": 0, "cam2": 0, "cam3": None, "cam4": -1, "cam5": -2, "cam6": -5, "cam7": -7}
    assert r["cameras"]["cam3"]["status"] == "no_flash"
    assert abs(r["cameras"]["cam5"]["offset_ms"] - (-2 * 1000 / 30)) < 1e-6


def test_reference_falls_back_when_cam0_has_no_flash():
    ev = {k: v for k, v in TAKE1_EVENTS.items()}
    ev["cam0"] = []
    ev["cam1"] = [(2, 66.8), (36, 31.1)]   # with cam0 gone, cam1's extra 32 would be a genuine tie against its first flash
    r = fo.compute_offsets(ev, CAMS, 30.0)
    assert r["reference"] == "cam1"
    assert r["cameras"]["cam1"]["offset_frames"] == 0
    assert r["cameras"]["cam7"]["offset_frames"] == -7


def test_lone_event_is_not_trusted():
    ev = {"cam0": [(35, 40.0)], "cam1": [(33, 40.0)], "cam2": [(35, 40.0)]}
    r = fo.compute_offsets(ev, ["cam0", "cam1", "cam2"], 30.0)
    assert r["overall"] == "failed" and "double flash" in r["reason"]
    assert fo.start_frames_from(r, ["cam0", "cam1", "cam2"]) == [0, 0, 0]


def test_double_flash_with_one_camera_seeing_only_the_white_flash():
    ev = {"cam0": [(3, 50.0), (36, 40.0)], "cam1": [(2, 50.0), (35, 40.0)], "cam2": [(36, 40.0)]}
    r = fo.compute_offsets(ev, ["cam0", "cam1", "cam2"], 30.0)
    assert r["overall"] == "ok"
    assert [r["cameras"][c]["offset_frames"] for c in ("cam0", "cam1", "cam2")] == [0, -1, 0]


def test_no_flash_is_failed():
    r = fo.compute_offsets({"cam0": [], "cam1": [], "cam2": []}, ["cam0", "cam1", "cam2"], 30.0)
    assert r["overall"] == "failed" and r["reference"] is None
    assert fo.start_frames_from(r, ["cam0", "cam1", "cam2"]) == [0, 0, 0]


def test_only_one_camera_with_flash_is_failed():
    r = fo.compute_offsets({"cam0": [(35, 40.0)], "cam1": [], "cam2": []}, ["cam0", "cam1", "cam2"], 30.0)
    assert r["overall"] == "failed"


def test_camera_with_flashes_far_from_the_others_is_not_aligned():
    ev = {c: [(3, 50.0), (36, 40.0)] for c in ("cam0", "cam1", "cam2")}
    ev["cam3"] = [(23, 50.0), (56, 40.0)]       # a double flash with the right spacing, but 20 frames away from everyone else
    r = fo.compute_offsets(ev, ["cam0", "cam1", "cam2", "cam3"], 30.0)
    assert r["cameras"]["cam3"]["offset_frames"] is None and r["cameras"]["cam3"]["status"] != "ok"
    assert r["overall"] == "partial"


def test_start_frames_are_non_negative_and_zero_for_unusable():
    r = fo.compute_offsets(TAKE1_EVENTS, CAMS, 30.0)
    sf = fo.start_frames_from(r, CAMS)
    assert min(sf) == 0 and all(n >= 0 for n in sf)
    assert sf == [7, 7, 7, 0, 6, 5, 2, 0]   # offset - min(offset); cam3 (no flash) stays unshifted


def test_overlay_png_has_banner_only_when_not_ok(tmp_path):
    ok = fo.compute_offsets({c: [(3, 50.0), (36, 40.0)] for c in ("cam0", "cam1")}, ["cam0", "cam1"], 30.0)
    bad = fo.compute_offsets({"cam0": [], "cam1": []}, ["cam0", "cam1"], 30.0)
    for name, res, banner in (("ok", ok, False), ("bad", bad, True)):
        path = fo.build_overlay_png(res, ["cam0", "cam1"], 640, 360, str(tmp_path / f"{name}.png"))
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        assert img.shape == (360, 1280, 4)
        assert bool(img[5, 600, 3] > 0) == banner   # top bar is opaque only on failure


def test_flash_in_a_minority_of_cameras_is_failed():
    cams = [f"cam{i}" for i in range(7)]
    ev = {c: [] for c in cams}
    ev["cam3"], ev["cam4"] = [(103, 50.0), (134, 40.0)], [(100, 50.0), (131, 40.0)]   # only 2 of 7 cameras saw the double flash
    r = fo.compute_offsets(ev, cams, 30.0)
    assert r["overall"] == "failed" and "2 of 7" in r["reason"]
    assert fo.start_frames_from(r, cams) == [0] * 7


def test_flash_sync_summary_and_status_text():
    import postprocess as pp
    r = fo.compute_offsets(TAKE1_EVENTS, CAMS, 30.0)
    fs = fo.flash_sync_summary(r)
    assert fs["overall"] == "partial" and fs["missing"] == ["cam3"] and fs["n_ok"] == 7 and fs["n_total"] == 8
    text = pp.status_text({"status": "done", "flash_sync": fs})
    assert "FLASH SYNC PARTIAL" in text and "cam3" in text and "⚠️" in text
    ok = fo.compute_offsets({c: [(3, 50.0), (36, 40.0)] for c in ("cam0", "cam1")}, ["cam0", "cam1"], 30.0)
    t_ok = pp.status_text({"status": "done", "flash_sync": fo.flash_sync_summary(ok)})
    assert "flash sync OK, 2/2" in t_ok and "⚠️" not in t_ok
    bad = fo.flash_sync_summary(fo.compute_offsets({"cam0": [], "cam1": []}, ["cam0", "cam1"], 30.0))
    assert "FLASH SYNC FAILED" in pp.status_text({"status": "done", "flash_sync": bad})
    assert pp.status_text({"status": "done"}) == "postprocessed"          # take postprocessed before flash sync existed
    assert pp.status_text({"status": "failed", "error": "boom"}) == "postprocessing failed (boom)"


def test_status_file_carries_flash_sync_and_offsets_load_back(tmp_path):
    import postprocess as pp
    take = str(tmp_path)
    r = fo.compute_offsets(TAKE1_EVENTS, CAMS, 30.0)
    fs = fo.flash_sync_summary(r)
    pp._write_status(take, "done", extra={"flash_sync": fs})
    assert pp.read_status(take) == {"status": "done", "flash_sync": fs}
    assert fo.load_flash_offsets(take) is None                              # nothing saved yet
    fo.write_outputs(r, {}, os.path.join(take, "processed"))
    loaded = fo.load_flash_offsets(take)
    assert loaded["offsets"]["cam7"] == -7 and loaded["offsets"]["cam3"] is None
    assert loaded["start_frames"] == dict(zip(CAMS, [7, 7, 7, 0, 6, 5, 2, 0])) and loaded["usable"]


def test_sixty_fps_scales_the_time_based_limits():
    # same physical timing at 60 fps: RGB cycle at 0.6 s, white flash at 1.7 s, gap 1.1 s = 66 frames, offsets double
    ev = {"cam0": [(36, 30.0), (102, 25.0)], "cam1": [(38, 30.0), (104, 25.0)], "cam2": [(34, 30.0), (100, 25.0)]}
    r = fo.compute_offsets(ev, ["cam0", "cam1", "cam2"], 60.0)
    assert r["overall"] == "ok"
    assert [r["cameras"][c]["offset_frames"] for c in ("cam0", "cam1", "cam2")] == [0, 2, -2]
    assert abs(r["cameras"]["cam1"]["offset_ms"] - 2 * 1000 / 60) < 1e-6


def test_weak_camera_is_rescued_only_with_the_expected_spacing():
    ev = {c: [(36, 30.0), (102, 25.0)] for c in ("cam0", "cam1", "cam2")}
    cams = ["cam0", "cam1", "cam2", "cam3", "cam4"]
    weak = {"cam3": [(37, 12.0), (103, 11.0)],     # right spacing (66 frames) -> accepted, flagged weak
            "cam4": [(37, 12.0), (90, 11.0)]}      # onset in both windows' neighbourhood but wrong spacing / out of window -> rejected
    r = fo.compute_offsets(ev, cams, 60.0, weak_events=weak)
    assert r["cameras"]["cam3"]["status"] == "weak" and r["cameras"]["cam3"]["offset_frames"] == 1
    assert r["cameras"]["cam4"]["offset_frames"] is None
    assert fo.flash_sync_summary(r)["check"] == ["cam3"]


# ---- calibration-guided pass (flash_guided.py) ----
import json  # noqa: E402
import flash_guided as fg  # noqa: E402


def _ring_of_cameras(n=6, width=2016, f=1150.0):
    """n synthetic cameras on a circle of radius 2 m looking at the origin (distortion-free)."""
    cams = {}
    for i in range(n):
        a = 2 * np.pi * i / n
        pos = np.array([2 * np.cos(a), 0.0, 2 * np.sin(a)])
        z = -pos / np.linalg.norm(pos)
        x = np.cross([0, 1, 0], z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])
        cams[f"cam{i}"] = {"K": np.array([[f, 0, width / 2], [0, f, 760.0], [0, 0, 1]]), "dist": np.zeros(5), "R": R,
                           "t": -R @ pos, "width": width, "height": 1520, "device_id": f"dev{i}"}
    return cams


def test_triangulate_ignores_one_camera_on_clutter():
    cams = _ring_of_cameras()
    X = np.array([0.1, -0.05, 0.2])
    pts = {c: tuple(fg.project(X, v)) for c, v in cams.items()}
    pts["cam3"] = (2000.0, 100.0)                       # locked onto clutter
    tri = fg.triangulate(pts, cams, 2016)
    assert tri is not None and "cam3" not in tri["inliers"] and len(tri["inliers"]) == 5
    assert np.allclose(tri["X"], X, atol=1e-3)
    assert tri["errors"]["cam3"] > 100


def test_triangulate_needs_three_agreeing_cameras():
    cams = _ring_of_cameras(4)
    pts = {"cam0": (500.0, 500.0), "cam1": (1500.0, 900.0), "cam2": (300.0, 1200.0)}   # three unrelated pixels
    assert fg.triangulate(pts, cams, 2016) is None


def _guided_base():
    cams = {c: {"flash_frame": 100, "cross_check_frame": 34, "offset_frames": 0, "offset_ms": 0.0, "ts_implied_offset_ms": None,
                "residual_ms": None, "status": "ok", "reason": ""} for c in ("cam0", "cam1", "cam2", "cam3")}
    cams["cam2"].update(status="weak", flash_frame=101, offset_frames=1)
    cams["cam3"].update(status="no_flash", flash_frame=None, offset_frames=None, offset_ms=None, reason="no flash found")
    return {"fps": 60.0, "reference": "cam0", "overall": "partial", "reason": "no usable flash in cam3", "cameras": cams}


def test_apply_guided_fills_missing_and_keeps_detector_value():
    r = _guided_base()
    onsets = {"cam0": {"onset": 100, "rise": 30}, "cam1": {"onset": 99, "rise": 30}, "cam2": {"onset": 99, "rise": 25},
              "cam3": {"onset": 101, "rise": 20}}
    assert fo.apply_guided(r, onsets)
    off = {c: (v["offset_frames"], v["status"], v["detector_offset_frames"]) for c, v in r["cameras"].items()}
    assert off == {"cam0": (0, "ok", 0), "cam1": (-1, "ok", 0), "cam2": (-1, "guided", 1), "cam3": (1, "guided", None)}
    assert r["overall"] == "ok"
    assert fo.flash_sync_summary(r)["check"] == ["cam2", "cam3"]


def test_apply_guided_leaves_cameras_without_a_flash_at_the_projected_position():
    r = _guided_base()
    onsets = {"cam0": {"onset": 100, "rise": 30}, "cam1": {"onset": 100, "rise": 30}, "cam2": {"onset": None, "rise": 3},
              "cam3": {"onset": None, "rise": 2}}
    assert fo.apply_guided(r, onsets)
    assert r["cameras"]["cam2"]["offset_frames"] == 1 and r["cameras"]["cam2"]["status"] == "weak"
    assert r["cameras"]["cam3"]["offset_frames"] is None and "position triangulated" in r["cameras"]["cam3"]["reason"]
    assert r["overall"] == "partial"


def test_apply_guided_needs_the_reference_to_show_the_flash():
    r = _guided_base()
    assert not fo.apply_guided(r, {"cam0": {"onset": None, "rise": 2}, "cam1": {"onset": 100, "rise": 30}})
    assert r["cameras"]["cam1"]["offset_frames"] == 0 and "detector_offset_frames" not in r["cameras"]["cam1"]


def test_find_calibration_picks_the_closest_file_with_matching_width_and_devices(tmp_path):
    cams = _ring_of_cameras(2)

    def write(name, ts, width, dev=None):
        payload = {"metadata": {"timestamp": ts}, "cameras": {c: {
            "device_id": dev or v["device_id"],
            "intrinsics": {"camera_matrix": v["K"].tolist(), "dist_coeffs": [0] * 5, "image_width": width, "image_height": 1520},
            "extrinsics": {"rotation": v["R"].tolist(), "translation": v["t"].tolist()}} for c, v in cams.items()}}
        (tmp_path / name).write_text(json.dumps(payload))

    write("a.json", "2026-10-22T17:00:00", 2016)
    write("b.json", "2026-10-22T17:30:00", 2016)          # closest to the take (17:19)
    write("c.json", "2026-10-22T17:20:00", 3840)          # wrong width
    write("d.json", "2026-10-22T17:19:30", 2016, "other")   # other devices
    take = tmp_path / "take"
    take.mkdir()
    (take / "take_meta.json").write_text(json.dumps({"start_time_iso": "2026-10-22T17:19:05",
                                                     "cameras": {c: {"device_id": v["device_id"]} for c, v in cams.items()}}))
    path, found = fg.find_calibration(str(take), list(cams), 2016, dirs=[str(tmp_path)])
    assert os.path.basename(path) == "b.json" and set(found) == set(cams)
    none, why = fg.find_calibration(str(take), list(cams), 1000, dirs=[str(tmp_path)])
    assert none is None and "no calibration" in why
