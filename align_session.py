"""Post-capture timestamp alignment for multi-camera OAK sessions."""

import argparse
import bisect
import json
import math
import os
import re
import shutil
import subprocess
import glob
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm

PREVIEW_GRID_W = 1920
PREVIEW_GRID_H = 1080
BLUE_BGR = (255, 0, 0)


def preview_grid_layout(num_cameras):
    cols = math.ceil(math.sqrt(num_cameras))
    rows = math.ceil(num_cameras / cols)
    return cols, rows


def xstack_layout(num_cameras):
    cols, _rows = preview_grid_layout(num_cameras)
    parts = []
    for i in range(num_cameras):
        row, col = divmod(i, cols)
        x = "0" if col == 0 else f"w{i - 1}"
        y = "0" if row == 0 else f"h{i - cols}"
        parts.append(f"{x}_{y}")
    return "|".join(parts)


def grid_mp4_path(session_dir):
    return os.path.join(session_dir, "compressed_video_grid.mp4")


def default_align_threshold_ms(fps):
    return 0.5 / fps * 1000.0


def build_slot_times(parsed, cameras, time_key, fps):
    """Slot times = reference camera (cam0) frame timestamps within overlap."""
    t_start = max(entries[0][time_key] for entries in parsed.values())
    t_end = min(entries[-1][time_key] for entries in parsed.values())
    if t_end <= t_start:
        raise ValueError("No overlapping window across cameras")

    ref_label = cameras[0]["label"]
    slot_times = [
        e[time_key] for e in parsed[ref_label]
        if t_start <= e[time_key] <= t_end + 1e-9
    ]
    if not slot_times:
        raise ValueError(f"No reference slots from {ref_label} in overlap window")
    return t_start, t_end, slot_times, ref_label


def discover_session(session_dir):
    session_dir = os.path.abspath(session_dir)
    if not os.path.isdir(session_dir):
        raise FileNotFoundError(f"Session folder not found: {session_dir}")

    base = os.path.basename(session_dir)
    if not base.startswith("sign_capture_"):
        print(f"[Alignment] Note: folder name '{base}' does not match sign_capture_* convention")

    video_paths = sorted(glob.glob(os.path.join(session_dir, "video_cam*.mjpeg")))
    if not video_paths:
        raise FileNotFoundError(f"No video_cam*.mjpeg files in {session_dir}")

    cameras = []
    for video_path in video_paths:
        name = os.path.basename(video_path)
        cam_label = name.removeprefix("video_").removesuffix(".mjpeg")
        log_path = os.path.join(session_dir, f"frame_timestamps_{cam_label}.log")
        if not os.path.isfile(log_path):
            raise FileNotFoundError(f"Missing log for {cam_label}: {log_path}")
        cameras.append({
            "label": cam_label,
            "video_path": video_path,
            "log_path": log_path,
        })

    return session_dir, cameras


def parse_host_ts_s(value):
    try:
        return float(value)
    except ValueError:
        return datetime.fromisoformat(value).timestamp()


def parse_log_header(log_path):
    """Parse log header for resolution, fps, and optional sync calibration."""
    rec_w, rec_h, fps = None, None, None
    host_offset_s = None
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                break
            m = re.search(r"(\d+)x(\d+)@(\d+(?:\.\d+)?)fps", line)
            if m:
                rec_w, rec_h, fps = int(m.group(1)), int(m.group(2)), float(m.group(3))
            if line.startswith("# sync_calibration"):
                offset_m = re.search(r"host_offset_s=([-\d.]+)", line)
                if offset_m:
                    host_offset_s = float(offset_m.group(1))
    return rec_w, rec_h, fps, host_offset_s


def parse_timestamp_log(log_path, host_offset_s=None):
    entries = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            host_ts_s = parse_host_ts_s(parts[0])
            seq = int(parts[1])
            device_ts_s = float(parts[2])
            entry = {
                "idx": len(entries),
                "seq": seq,
                "device_ts_s": device_ts_s,
                "host_ts_s": host_ts_s,
            }
            if host_offset_s is not None:
                entry["unified_ts_s"] = device_ts_s + host_offset_s
            entries.append(entry)
    return entries


def nearest_entry(entries, slot_t, time_key="unified_ts_s"):
    if not entries:
        return None, None
    times = [e[time_key] for e in entries]
    i = bisect.bisect_left(times, slot_t)
    candidates = []
    if i > 0:
        candidates.append(i - 1)
    if i < len(times):
        candidates.append(i)
    best = min(candidates, key=lambda j: abs(times[j] - slot_t))
    return entries[best], abs(times[best] - slot_t)


def read_mjpeg_frame(video_path, frame_idx, cache, cap_holder):
    if frame_idx in cache:
        return cache[frame_idx]
    if cap_holder[0] is None:
        cap_holder[0] = cv2.VideoCapture(video_path)
        if not cap_holder[0].isOpened():
            raise RuntimeError(f"Failed to open MJPEG: {video_path}")
    cap = cap_holder[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    cache[frame_idx] = frame
    return frame


def make_blue_frame(rec_w, rec_h):
    img = np.zeros((rec_h, rec_w, 3), dtype=np.uint8)
    img[:] = BLUE_BGR
    return img


def percentile(values, pct):
    if not values:
        return 0.0
    arr = np.array(values)
    return float(np.percentile(arr, pct))


def align_session(
    session_dir,
    *,
    align_threshold_ms=None,
    fps=None,
    rec_w=None,
    rec_h=None,
    align_host_only=False,
):
    session_dir, cameras = discover_session(session_dir)

    header_w, header_h, header_fps, _ = parse_log_header(cameras[0]["log_path"])
    rec_w = rec_w or header_w or 3840
    rec_h = rec_h or header_h or 2160
    fps = fps or header_fps or 30.0
    threshold_s = (align_threshold_ms if align_threshold_ms is not None
                   else default_align_threshold_ms(fps)) / 1000.0
    align_threshold_ms = threshold_s * 1000.0

    offsets = {}
    for cam in cameras:
        _w, _h, _fps, host_offset_s = parse_log_header(cam["log_path"])
        offsets[cam["label"]] = host_offset_s

    if align_host_only:
        time_key = "host_ts_s"
        timebase = "host"
        use_unified = False
    elif all(offset is not None for offset in offsets.values()):
        time_key = "unified_ts_s"
        timebase = "unified_device"
        use_unified = True
    else:
        missing = [label for label, off in offsets.items() if off is None]
        print(
            "[Alignment] WARNING: missing sync_calibration in "
            f"{', '.join(missing)}; falling back to raw host timestamps"
        )
        time_key = "host_ts_s"
        timebase = "host_legacy"
        use_unified = False

    parsed = {
        cam["label"]: parse_timestamp_log(
            cam["log_path"],
            host_offset_s=offsets[cam["label"]] if use_unified else None,
        )
        for cam in cameras
    }
    for cam in cameras:
        label = cam["label"]
        if not parsed[label]:
            raise ValueError(f"No frame entries in log for {label}")

    t_start, t_end, slot_times, ref_label = build_slot_times(
        parsed, cameras, time_key, fps
    )
    num_slots = len(slot_times)
    frame_period_ms = 1000.0 / fps
    if align_threshold_ms < frame_period_ms * 0.45:
        print(
            f"[Alignment] WARNING: threshold {align_threshold_ms:.1f}ms is below "
            f"~half a frame ({frame_period_ms / 2:.1f}ms); expect more blue slots"
        )
    duration_s = num_slots / fps
    cam_labels = [cam["label"] for cam in cameras]
    t_start_iso = datetime.fromtimestamp(t_start).isoformat(timespec="milliseconds")
    t_end_iso = datetime.fromtimestamp(t_end).isoformat(timespec="milliseconds")
    print(f"\n[Alignment] Session: {os.path.basename(session_dir)}")
    print(f"[Alignment] Timebase: {timebase}")
    print(f"[Alignment] Cameras: {', '.join(cam_labels)}")
    print(f"[Alignment] Slot times: {num_slots} reference frames from {ref_label}")
    if use_unified:
        for label, off in offsets.items():
            print(f"[Alignment]   {label}: host_offset_s={off:.9f}")
    print(
        f"[Alignment] Timeline: {num_slots} slots @ {fps:.1f} FPS "
        f"({duration_s:.1f}s wall time)"
    )
    print(
        f"[Alignment] Overlap window: {t_start_iso} -> {t_end_iso} "
        f"| threshold {align_threshold_ms:.1f}ms | {rec_w}x{rec_h}"
    )
    for cam in cameras:
        label = cam["label"]
        print(f"[Alignment]   {label}: {len(parsed[label])} logged frames")

    aligned_dir = os.path.join(session_dir, "aligned")
    if os.path.isdir(aligned_dir):
        print(f"[Alignment] Overwriting existing folder: {aligned_dir}")
        shutil.rmtree(aligned_dir)

    out_dirs = {}
    for cam in cameras:
        out_dir = os.path.join(aligned_dir, cam["label"])
        os.makedirs(out_dir, exist_ok=True)
        out_dirs[cam["label"]] = out_dir

    blue = make_blue_frame(rec_w, rec_h)
    decode_cache = {cam["label"]: {} for cam in cameras}
    cap_holders = {cam["label"]: [None] for cam in cameras}
    slot_matches = {cam["label"]: [] for cam in cameras}
    matched_total = 0
    blue_total = 0
    cross_camera_spreads_ms = []

    slot_iter = tqdm(
        enumerate(slot_times),
        total=num_slots,
        unit="slot",
        desc="Aligning",
        dynamic_ncols=True,
    )
    for slot_idx, slot_t in slot_iter:
        proposals = {}
        for cam in cameras:
            label = cam["label"]
            entry, delta = nearest_entry(parsed[label], slot_t, time_key=time_key)
            proposals[label] = (entry, delta)

        slot_ok = all(
            entry is not None and delta <= threshold_s
            for entry, delta in proposals.values()
        )

        slot_matched_times = {}
        for cam in cameras:
            label = cam["label"]
            entry, delta = proposals[label]
            out_path = os.path.join(out_dirs[label], f"{slot_idx:06d}.jpg")

            if not slot_ok:
                if entry is None:
                    status = "missing"
                elif delta > threshold_s:
                    status = "discarded"
                else:
                    status = "discarded_slot"
                info = {"slot": slot_idx, "status": status}
                if entry is not None and delta > threshold_s:
                    info["delta_ms"] = delta * 1000.0
                    info["nearest_idx"] = entry["idx"]
                elif entry is not None and status == "discarded_slot":
                    info["reason"] = "another_camera_out_of_threshold"
                    info["nearest_idx"] = entry["idx"]
                slot_matches[label].append(info)
                cv2.imwrite(out_path, blue)
                blue_total += 1
                continue

            frame = read_mjpeg_frame(
                cam["video_path"], entry["idx"],
                decode_cache[label], cap_holders[label],
            )
            if frame is None:
                cv2.imwrite(out_path, blue)
                slot_matches[label].append({"slot": slot_idx, "status": "missing_decode"})
                blue_total += 1
                continue

            cv2.imwrite(out_path, frame)
            matched_total += 1
            offset_ms = (entry[time_key] - slot_t) * 1000.0
            match_info = {
                "slot": slot_idx,
                "status": "matched",
                "idx": entry["idx"],
                "offset_ms": offset_ms,
            }
            if time_key == "unified_ts_s":
                match_info["unified_ts_s"] = entry["unified_ts_s"]
            else:
                match_info["host_ts_s"] = entry["host_ts_s"]
            slot_matches[label].append(match_info)
            slot_matched_times[label] = entry[time_key]

        if len(slot_matched_times) == len(cameras):
            spread_ms = (
                (max(slot_matched_times.values()) - min(slot_matched_times.values()))
                * 1000.0
            )
            cross_camera_spreads_ms.append(spread_ms)

        if slot_idx % 25 == 0 or slot_idx == num_slots - 1:
            slot_iter.set_postfix(matched=matched_total, blue=blue_total, refresh=False)

    for cap_holder in cap_holders.values():
        if cap_holder[0] is not None:
            cap_holder[0].release()

    print(f"[Alignment] Wrote {num_slots * len(cameras)} JPEGs to {aligned_dir}")

    frame_period_ms = 1000.0 / fps
    cross_spread_mean = float(np.mean(cross_camera_spreads_ms)) if cross_camera_spreads_ms else 0.0
    cross_spread_max = float(max(cross_camera_spreads_ms)) if cross_camera_spreads_ms else 0.0
    cross_spread_p95 = percentile(cross_camera_spreads_ms, 95)

    spread_key = (
        "cross_camera_unified_spread_ms"
        if use_unified
        else "cross_camera_host_spread_ms"
    )
    overlap_start_key = (
        "overlap_start_unified_s" if use_unified else "overlap_start_host_s"
    )
    overlap_end_key = (
        "overlap_end_unified_s" if use_unified else "overlap_end_host_s"
    )

    report = {
        "session_dir": session_dir,
        "timebase": timebase,
        "fps": fps,
        "rec_w": rec_w,
        "rec_h": rec_h,
        "align_threshold_ms": align_threshold_ms,
        "aligned_frame_count": len(slot_times),
        overlap_start_key: t_start,
        overlap_end_key: t_end,
        spread_key: {
            "mean": cross_spread_mean,
            "max": cross_spread_max,
            "p95": cross_spread_p95,
            "fully_matched_slots": len(cross_camera_spreads_ms),
        },
        "host_offsets_s": offsets if use_unified else {},
        "cameras": {},
        "warnings": [],
    }

    if cross_spread_p95 > frame_period_ms:
        report["warnings"].append(
            f"Cross-camera {timebase} spread p95 {cross_spread_p95:.1f}ms exceeds one frame "
            f"({frame_period_ms:.1f}ms)"
        )

    for cam in cameras:
        label = cam["label"]
        matches = slot_matches[label]
        matched = [m for m in matches if m["status"] == "matched"]
        discarded = [m for m in matches if m["status"] in ("discarded", "discarded_slot")]
        missing = [m for m in matches if m["status"] in ("missing", "missing_decode")]

        offsets_ms = [abs(m["offset_ms"]) for m in matched]
        mean_off = float(np.mean(offsets_ms)) if offsets_ms else 0.0
        max_off = float(max(offsets_ms)) if offsets_ms else 0.0
        p95_off = percentile(offsets_ms, 95)

        log_count = len(parsed[label])
        if len(matched) >= 20:
            n = max(1, len(matched) // 10)
            first_mean = float(np.mean([abs(m["offset_ms"]) for m in matched[:n]]))
            last_mean = float(np.mean([abs(m["offset_ms"]) for m in matched[-n:]]))
            if abs(last_mean - first_mean) > frame_period_ms:
                report["warnings"].append(
                    f"{label}: drift trend {first_mean:.1f}ms -> {last_mean:.1f}ms "
                    f"across session"
                )

        if max_off > align_threshold_ms and matched:
            report["warnings"].append(
                f"{label}: max offset {max_off:.1f}ms exceeds threshold "
                f"{align_threshold_ms:.1f}ms"
            )

        video_cap = cv2.VideoCapture(cam["video_path"])
        video_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if video_cap.isOpened() else -1
        video_cap.release()
        if video_count >= 0 and video_count != log_count:
            report["warnings"].append(
                f"{label}: log entries ({log_count}) != MJPEG frames ({video_count})"
            )

        report["cameras"][label] = {
            "matched_frames": len(matched),
            "discarded_frames": len(discarded),
            "missing_frames": len(missing),
            "mean_abs_offset_ms": mean_off,
            "max_abs_offset_ms": max_off,
            "p95_abs_offset_ms": p95_off,
            "blue_placeholders": len(discarded) + len(missing),
        }

    report_path = os.path.join(session_dir, "alignment_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_alignment_report(report)
    print(f"[Alignment] Report saved: {report_path}")
    return report


def print_alignment_report(report):
    duration = report["aligned_frame_count"] / report["fps"]
    timebase = report.get("timebase", "unified_device")
    spread = report.get("cross_camera_unified_spread_ms") or report.get(
        "cross_camera_host_spread_ms", {}
    )
    print(
        f"\n[Alignment] {report['aligned_frame_count']} aligned slots @ "
        f"{report['fps']:.1f} FPS ({duration:.1f}s) | "
        f"threshold {report['align_threshold_ms']:.1f}ms | "
        f"timebase {timebase}"
    )
    if spread:
        label = "unified" if timebase == "unified_device" else "host"
        if timebase == "host_legacy":
            label = "host (legacy)"
        print(
            f"[Alignment] Cross-camera {label} spread: "
            f"mean {spread.get('mean', 0):.1f}ms | "
            f"p95 {spread.get('p95', 0):.1f}ms | "
            f"max {spread.get('max', 0):.1f}ms "
            f"({spread.get('fully_matched_slots', 0)} fully matched slots)"
        )
    for label, stats in report["cameras"].items():
        print(
            f"  {label}: matched {stats['matched_frames']}, "
            f"discarded {stats['discarded_frames']}, "
            f"missing {stats['missing_frames']} | "
            f"offset mean {stats['mean_abs_offset_ms']:.1f}ms "
            f"max {stats['max_abs_offset_ms']:.1f}ms"
        )
        blue = stats["blue_placeholders"]
        if blue > 0:
            print(
                f"  {label}: {blue} blue placeholder frames "
                f"({stats['discarded_frames']} discarded, "
                f"{stats['missing_frames']} missing)"
            )
    for warning in report["warnings"]:
        print(f"[Alignment] WARNING: {warning}")



def convert_mjpegs_to_grid_mp4(video_paths, fps, session_dir):
    """Build preview grid MP4 directly from raw per-camera MJPEG files."""
    num_cameras = len(video_paths)
    cols, rows = preview_grid_layout(num_cameras)
    cell_w = PREVIEW_GRID_W // cols
    cell_h = PREVIEW_GRID_H // rows
    output_path = grid_mp4_path(session_dir)

    print(
        f"[Alignment] Building grid MP4 from raw MJPEG "
        f"({num_cameras} streams -> {output_path})..."
    )

    cmd = ["ffmpeg"]
    for path in video_paths:
        cmd.extend(["-f", "mjpeg", "-framerate", str(fps), "-i", path])

    scale_filters = [f"[{i}:v]scale={cell_w}:{cell_h}[v{i}]" for i in range(num_cameras)]
    stack_inputs = "".join(f"[v{i}]" for i in range(num_cameras))
    filter_complex = (
        ";".join(scale_filters)
        + f";{stack_inputs}xstack=inputs={num_cameras}:layout={xstack_layout(num_cameras)}[v]"
    )

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-c:v", "libx264",
        "-r", str(fps),
        "-crf", "18",
        "-preset", "medium",
        "-tune", "film",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-y", output_path,
    ])
    subprocess.run(cmd, check=True)
    return output_path


def export_raw_grid_mp4(session_dir, fps=None):
    """Skip alignment; write grid MP4 from session MJPEG files."""
    session_dir, cameras = discover_session(session_dir)
    header_w, header_h, header_fps, _ = parse_log_header(cameras[0]["log_path"])
    fps = fps or header_fps or 30.0
    video_paths = [cam["video_path"] for cam in cameras]
    cam_labels = [cam["label"] for cam in cameras]
    print("\n[Alignment] Raw export (no timestamp alignment)")
    print(f"[Alignment] Session: {os.path.basename(session_dir)}")
    print(f"[Alignment] Cameras: {', '.join(cam_labels)} @ {fps:.1f} FPS")
    return convert_mjpegs_to_grid_mp4(video_paths, fps, session_dir)


def convert_aligned_jpegs_to_grid_mp4(session_dir, cam_labels, fps):
    num_cameras = len(cam_labels)
    cols, rows = preview_grid_layout(num_cameras)
    cell_w = PREVIEW_GRID_W // cols
    cell_h = PREVIEW_GRID_H // rows
    aligned_dir = os.path.join(session_dir, "aligned")
    output_path = grid_mp4_path(session_dir)

    print(f"[Alignment] Building grid MP4 ({num_cameras} streams -> {output_path})...")

    cmd = ["ffmpeg"]
    for label in cam_labels:
        pattern = os.path.join(aligned_dir, label, "%06d.jpg")
        cmd.extend(["-framerate", str(fps), "-i", pattern])

    scale_filters = [f"[{i}:v]scale={cell_w}:{cell_h}[v{i}]" for i in range(num_cameras)]
    stack_inputs = "".join(f"[v{i}]" for i in range(num_cameras))
    filter_complex = (
        ";".join(scale_filters)
        + f";{stack_inputs}xstack=inputs={num_cameras}:layout={xstack_layout(num_cameras)}[v]"
    )

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-c:v", "libx264",
        "-r", str(fps),
        "-crf", "18",
        "-preset", "medium",
        "-tune", "film",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-y", output_path,
    ])
    subprocess.run(cmd, check=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Align multi-camera session using unified device timestamps"
    )
    parser.add_argument("session_dir", help="Path to session folder")
    parser.add_argument(
        "--align-threshold-ms",
        type=float,
        default=None,
        help="Max timestamp offset to accept a match (default: half frame period)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override FPS (default: parse from log header)",
    )
    parser.add_argument(
        "--output-mp4",
        choices=["small"],
        default=None,
        help="Build grid MP4 from aligned JPEGs",
    )
    parser.add_argument(
        "--align-host",
        action="store_true",
        help="Align on raw host receive timestamps (ignore device sync calibration)",
    )
    parser.add_argument(
        "--align-raw",
        action="store_true",
        help="Skip alignment; build grid MP4 from raw MJPEG only",
    )
    args = parser.parse_args()

    if args.align_raw:
        if args.align_host:
            parser.error("--align-raw cannot be combined with --align-host")
        mp4_path = export_raw_grid_mp4(args.session_dir, fps=args.fps)
        print(f"[Success] Grid mp4 saved: {mp4_path}")
        return

    report = align_session(
        args.session_dir,
        align_threshold_ms=args.align_threshold_ms,
        fps=args.fps,
        align_host_only=args.align_host,
    )

    if args.output_mp4 == "small":
        cam_labels = list(report["cameras"].keys())
        mp4_path = convert_aligned_jpegs_to_grid_mp4(
            report["session_dir"], cam_labels, report["fps"]
        )
        print(f"[Success] Grid mp4 saved: {mp4_path}")


if __name__ == "__main__":
    main()
