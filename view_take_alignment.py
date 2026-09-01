"""Standalone viser viewer for reviewing a take's temporal alignment and
extrinsics stability: sync-offset and dropped-frame charts, a board-pose-
consistency check (when a calibration file is given), and the actual
aligned grid video. All the real analysis work happens in
align_session.py / alignment_analysis.py -- this script only calls those
and renders the results, so it can later become a panel inside app.py's
own postprocess view without a rewrite.

    python view_take_alignment.py <take_dir> [--calibration PATH] [--force-realign]
                                   [--port 8080] [--stride-slots N] [--target-hz F]
"""
import argparse
import threading
import time

import cv2
import numpy as np
import viser

import alignment_analysis

CAM_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
]


def _color_for(index):
    return CAM_COLORS[index % len(CAM_COLORS)]


def _build_offset_chart_data(report):
    cam_labels = sorted(report["cameras"].keys())
    num_slots = report["aligned_frame_count"]
    fps = report["fps"]
    x = np.arange(num_slots, dtype=np.float64) / fps
    arrays = [x]
    for label in cam_labels:
        offsets = report["cameras"][label]["slot_offsets_ms"]
        y = np.array([v if v is not None else np.nan for v in offsets], dtype=np.float64)
        arrays.append(y)
    return cam_labels, tuple(arrays)


def _build_frame_shift_chart_data(report):
    """y = source_idx - slot_idx per slot per camera (nan at blue-placeholder
    slots). This is the camera's own raw frame index drifting away from a
    naive 1:1 "slot N == source frame N" mapping -- positive means the
    frame actually used at that slot came from later in this camera's own
    stream than the naive mapping (frames had to be skipped ahead to keep
    the camera in sync), negative means it came from earlier (frames held/
    repeated to wait for the reference grid). The reference camera itself
    should sit at (or very near) zero throughout, since slot times are
    built from its own frame timestamps.
    """
    cam_labels = sorted(report["cameras"].keys())
    num_slots = report["aligned_frame_count"]
    fps = report["fps"]
    x = np.arange(num_slots, dtype=np.float64) / fps
    arrays = [x]
    for label in cam_labels:
        src = report["cameras"][label]["slot_source_idx"]
        y = np.array(
            [(v - slot) if v is not None else np.nan for slot, v in enumerate(src)],
            dtype=np.float64,
        )
        arrays.append(y)
    return cam_labels, tuple(arrays)


def _build_frame_step_chart_data(report):
    """y = idx[n] - idx[n-1] between consecutive MATCHED slots for a camera
    (skipping over blue-placeholder slots in between, since those carry no
    source frame at all). 1 = normal forward playback, 0 = the previous
    frame was held/repeated (no new source frame had arrived yet), >1 =
    frames were skipped ahead (source ran fast relative to the reference),
    <1 (incl. negative) = a frame moved backward relative to the previous
    pick -- an anomaly, not expected in normal operation. Plotted at the
    slot where the step "arrives".
    """
    cam_labels = sorted(report["cameras"].keys())
    num_slots = report["aligned_frame_count"]
    fps = report["fps"]
    x = np.arange(num_slots, dtype=np.float64) / fps
    arrays = [x]
    for label in cam_labels:
        src = report["cameras"][label]["slot_source_idx"]
        y = np.full(num_slots, np.nan, dtype=np.float64)
        prev_idx = None
        for slot, v in enumerate(src):
            if v is None:
                continue
            if prev_idx is not None:
                y[slot] = v - prev_idx
            prev_idx = v
        arrays.append(y)
    return cam_labels, tuple(arrays)


def _build_gap_chart_data(report):
    """x-axis here is each camera's own raw log frame index -- NOT the
    same "seconds into take" axis as the offset chart, since a gap is a
    property of one camera's own sequence_num stream, not of the shared
    aligned-slot grid. Falls back to aligned_frame_count as the axis
    length when there are no gaps at all (a flat zero line), so the chart
    is never degenerate.
    """
    cam_labels = sorted(report["cameras"].keys())
    all_gaps = {label: report["cameras"][label]["sequence_gaps"]["gaps"] for label in cam_labels}
    max_idx = max(
        (g["after_idx"] for gaps in all_gaps.values() for g in gaps),
        default=report["aligned_frame_count"] - 1,
    )
    length = max_idx + 1
    x = np.arange(length, dtype=np.float64)
    arrays = [x]
    for label in cam_labels:
        y = np.zeros(length, dtype=np.float64)
        for g in all_gaps[label]:
            if g["after_idx"] < length:
                y[g["after_idx"]] = g["dropped_frames"]
        arrays.append(y)
    return cam_labels, tuple(arrays)


def _build_self_consistency_chart_data(report, board_report):
    num_slots = report["aligned_frame_count"]
    fps = report["fps"]
    stride = board_report["stride_slots"]
    sample_slots = list(range(0, num_slots, stride))
    x = np.array(sample_slots, dtype=np.float64) / fps
    cam_labels = sorted(board_report["per_camera_self_consistency"].keys())
    arrays = [x]
    for label in cam_labels:
        by_slot = {s["slot"]: s for s in board_report["per_camera_self_consistency"][label]["samples"]}
        if by_slot:
            mean_t = np.mean([s["t_world"] for s in by_slot.values()], axis=0)
        else:
            mean_t = None
        y = []
        for slot in sample_slots:
            s = by_slot.get(slot)
            if s is None or mean_t is None:
                y.append(np.nan)
            else:
                y.append(float(np.linalg.norm(np.array(s["t_world"]) - mean_t)))
        arrays.append(np.array(y, dtype=np.float64))
    return cam_labels, tuple(arrays)


def _build_cross_camera_chart_data(report, board_report):
    samples = board_report["cross_camera_consistency"]["samples"]
    if not samples:
        return None
    fps = report["fps"]
    x = np.array([s["slot"] for s in samples], dtype=np.float64) / fps
    y_trans = np.array([s["mean_translation_disagreement_m"] for s in samples], dtype=np.float64)
    y_rot = np.array([s["mean_rotation_disagreement_deg"] for s in samples], dtype=np.float64)
    return x, y_trans, y_rot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("take_dir")
    parser.add_argument("--calibration", default=None,
                         help="Calibration JSON to match cameras against by device_id for the "
                              "board-consistency check. Omit to skip that panel.")
    parser.add_argument("--force-realign", action="store_true",
                         help="Re-run align_session() even if alignment_report.json already exists.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--stride-slots", type=int, default=None)
    parser.add_argument("--target-hz", type=float, default=5.0)
    args = parser.parse_args()

    take_dir = args.take_dir

    print(f"[Viewer] Loading/building alignment report for {take_dir}...")
    report = alignment_analysis.ensure_alignment_report(take_dir, force=args.force_realign)
    print(f"[Viewer] Loading/building aligned grid video...")
    grid_video_path = alignment_analysis.ensure_aligned_grid_video(take_dir, report, force=args.force_realign)

    board_report = None
    if args.calibration:
        print(f"[Viewer] Running board-consistency check against {args.calibration}...")
        board_report = alignment_analysis.compute_board_consistency(
            take_dir, args.calibration, stride_slots=args.stride_slots,
            target_hz=args.target_hz, report=report,
        )
        alignment_analysis.save_board_consistency_report(take_dir, board_report)
        print(f"[Viewer]   cameras_used={list(board_report['cameras_used'].keys())} "
              f"cameras_skipped={list(board_report['cameras_skipped'].keys())}")

    server = viser.ViserServer(port=args.port)
    server.gui.add_markdown(f"# {take_dir}")
    server.gui.add_markdown(
        f"timebase: **{report['timebase']}** -- {report['aligned_frame_count']} aligned slots "
        f"@ {report['fps']} fps"
    )
    if report["warnings"]:
        server.gui.add_markdown("**Alignment warnings:**\n\n" + "\n".join(f"- {w}" for w in report["warnings"]))

    cam_labels_sorted = sorted(report["cameras"].keys())
    crop_lines = []
    any_crop = False
    for label in cam_labels_sorted:
        crop = report["cameras"][label]["crop"]
        if crop["before_frames"] or crop["after_frames"]:
            any_crop = True
        crop_lines.append(
            f"- {label}: {crop['before_frames']} frame(s) / {crop['before_s']:.2f}s cropped "
            f"from start, {crop['after_frames']} frame(s) / {crop['after_s']:.2f}s cropped "
            f"from end"
        )
    server.gui.add_markdown(
        "## Sequence cropping to align all views\n\n"
        "_Frames at the head/tail of each camera's own recording that fall outside the "
        "cross-camera overlap window -- i.e. footage this camera captured before the "
        "slowest-to-start camera began, or after the first-to-end camera stopped, "
        "and had to be thrown away so every camera covers the same aligned timeline._\n\n"
        + ("\n".join(crop_lines) if any_crop else "_No cropping was needed -- all cameras started/stopped within one frame of each other._")
    )

    server.gui.add_markdown("## Cross-reference offset (ms) over time")
    offset_labels, offset_data = _build_offset_chart_data(report)
    offset_series = [{}] + [
        {"label": label, "stroke": _color_for(i)} for i, label in enumerate(offset_labels)
    ]
    server.gui.add_uplot(
        data=offset_data, series=tuple(offset_series),
        axes=({"label": "seconds into take"}, {"label": "offset (ms)"}),
        scales={"x": {"time": False}},
        height=260,
    )

    server.gui.add_markdown(
        "## Frame-level adjustment\n\n"
        "_Every aligned slot is filled with a frame picked from each camera's own "
        "stream by nearest timestamp. These two charts show how far that pick had "
        "to move from a naive 1:1 mapping -- i.e. whether frames were held/repeated "
        "(camera lagging) or skipped ahead (camera running fast) to keep cameras in sync._"
    )
    shift_labels, shift_data = _build_frame_shift_chart_data(report)
    shift_series = [{}] + [
        {"label": label, "stroke": _color_for(i)} for i, label in enumerate(shift_labels)
    ]
    server.gui.add_markdown("### Frame index shift from nominal slot position (frames)")
    server.gui.add_uplot(
        data=shift_data, series=tuple(shift_series),
        axes=({"label": "seconds into take"}, {"label": "source idx - slot idx (frames)"}),
        scales={"x": {"time": False}},
        height=220,
    )

    step_labels, step_data = _build_frame_step_chart_data(report)
    step_series = [{}] + [
        {"label": label, "stroke": _color_for(i)} for i, label in enumerate(step_labels)
    ]
    server.gui.add_markdown(
        "### Step between consecutive used source frames\n\n"
        "_1 = normal forward playback. 0 = frame held/repeated. Above 1 = frame(s) "
        "skipped ahead. Below 1 (including negative) = frame moved backward -- "
        "unexpected, worth investigating._"
    )
    server.gui.add_uplot(
        data=step_data, series=tuple(step_series),
        axes=({"label": "seconds into take"}, {"label": "step (frames)"}),
        scales={"x": {"time": False}},
        height=220,
    )

    server.gui.add_markdown(
        "## Dropped frames (sequence_num gaps)\n\n"
        "_X-axis is each camera's own raw log frame index -- not the same "
        "timeline as the chart above, since a drop is a property of one "
        "camera's own frame stream, not the shared aligned-slot grid._"
    )
    gap_labels, gap_data = _build_gap_chart_data(report)
    gap_series = [{}] + [
        {"label": label, "stroke": _color_for(i)} for i, label in enumerate(gap_labels)
    ]
    server.gui.add_uplot(
        data=gap_data, series=tuple(gap_series),
        axes=({"label": "raw frame index"}, {"label": "dropped frames"}),
        scales={"x": {"time": False}},
        height=200,
    )

    if board_report is None:
        server.gui.add_markdown(
            "## Board-pose consistency\n\n_No --calibration passed -- pass a calibration JSON "
            "to check whether the stationary alignment board's pose agrees across cameras "
            "and stays stable over the take._"
        )
    else:
        skipped_lines = "\n".join(
            f"- {cam}: {reason}" for cam, reason in board_report["cameras_skipped"].items()
        )
        server.gui.add_markdown(
            f"## Board-pose consistency\n\n"
            f"Calibration: `{board_report['calibration_path']}`\n\n"
            f"Used: {', '.join(board_report['cameras_used']) or '_none_'}\n\n"
            + (f"Skipped:\n\n{skipped_lines}\n\n" if skipped_lines else "")
        )

        self_labels, self_data = _build_self_consistency_chart_data(report, board_report)
        if self_labels:
            server.gui.add_markdown("### Per-camera self-consistency (translation deviation from own mean, m)")
            self_series = [{}] + [
                {"label": label, "stroke": _color_for(i)} for i, label in enumerate(self_labels)
            ]
            server.gui.add_uplot(
                data=self_data, series=tuple(self_series),
                axes=({"label": "seconds into take"}, {"label": "deviation (m)"}),
                scales={"x": {"time": False}},
                height=220,
            )

        cross_data = _build_cross_camera_chart_data(report, board_report)
        if cross_data is not None:
            x, y_trans, y_rot = cross_data
            server.gui.add_markdown("### Cross-camera disagreement (mean translation, m)")
            server.gui.add_uplot(
                data=(x, y_trans), series=({}, {"label": "mean translation disagreement (m)", "stroke": "#e6194b"}),
                axes=({"label": "seconds into take"}, {"label": "disagreement (m)"}),
                scales={"x": {"time": False}},
                height=200,
            )
        else:
            server.gui.add_markdown(
                "### Cross-camera disagreement\n\n_No slot had 2+ cameras with usable "
                "intrinsics both detecting the board -- see cameras_skipped above._"
            )

    server.gui.add_markdown("## Aligned grid video")
    video_image = server.gui.add_image(np.zeros((4, 4, 3), dtype=np.uint8))
    video_status_md = server.gui.add_markdown("_Loading..._")

    def _playback_loop():
        cap = cv2.VideoCapture(grid_video_path)
        if not cap.isOpened():
            video_status_md.content = f"_Could not open `{grid_video_path}`_"
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or report["fps"]
        frame_period_s = 1.0 / fps
        video_status_md.visible = False
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            video_image.image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            time.sleep(frame_period_s)

    threading.Thread(target=_playback_loop, daemon=True).start()

    print(f"[Viewer] Ready at http://localhost:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
