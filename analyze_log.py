#!/usr/bin/env python3
"""analyze_log.py

Analyze capture session logs produced by capture.py.

Usage:
    python analyze_log.py path/to/session.log

Outputs a human-readable report with:
 - total frames, duration, expected frames (fps), measured fps
 - statistics on host timestamp intervals and device timestamp intervals
 - detected missing sequence numbers (dropped frames)
 - max/avg/median/std wait times
 - detection and optional trimming of bad start/end regions
"""
from __future__ import annotations
import sys
import re
from datetime import datetime
from typing import List, Tuple
import statistics


def parse_log(path: str):
    meta = {}
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # header metadata
                if line.startswith("# video="):
                    meta["video"] = line.split("=", 1)[1]
                elif "shutter_us" in line and "@" in line:
                    # example: # iso=200 shutter_us=2000 3840x2160@30fps mjpeg_q=95
                    m = re.search(r"(\d+)x(\d+)@(\d+)fps", line)
                    if m:
                        meta["res_w"] = int(m.group(1))
                        meta["res_h"] = int(m.group(2))
                        meta["fps"] = int(m.group(3))
                    m2 = re.search(r"iso=(\d+)", line)
                    if m2:
                        meta["iso"] = int(m2.group(1))
                    m3 = re.search(r"shutter_us=(\d+)", line)
                    if m3:
                        meta["shutter_us"] = int(m3.group(1))
                continue
            # data line: 2026-06-15T15:21:23.203119 0 1211280.774815000 734078
            parts = line.split()
            if len(parts) < 4:
                continue
            host_ts_s = parts[0]
            seq = int(parts[1])
            dev_ts = float(parts[2])
            size = int(parts[3])
            try:
                host_dt = datetime.fromisoformat(host_ts_s)
            except Exception:
                # fallback: try to parse without microseconds
                host_dt = datetime.strptime(host_ts_s, "%Y-%m-%dT%H:%M:%S")
            rows.append((host_dt, seq, dev_ts, size))
    return meta, rows


def diffs_from_timestamps(ts_list: List[float]) -> List[float]:
    return [t2 - t1 for t1, t2 in zip(ts_list[:-1], ts_list[1:])]


def detect_missing_sequences(seq_list: List[int]) -> Tuple[int, List[Tuple[int,int]]]:
    # returns total_missing, list of (prev_seq, next_seq) where gap occurred
    missing = 0
    gaps = []
    for a, b in zip(seq_list[:-1], seq_list[1:]):
        if b - a > 1:
            miss = (b - a - 1)
            missing += miss
            gaps.append((a, b))
    return missing, gaps


def stats(values: List[float]):
    if not values:
        return {}
    s = {}
    s["count"] = len(values)
    s["mean"] = statistics.mean(values)
    s["median"] = statistics.median(values)
    s["stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
    s["min"] = min(values)
    s["max"] = max(values)
    return s


def find_stable_region(intervals: List[float], expected: float, tol: float = 0.5, window:int = 5):
    """Return (start_index, end_index) inclusive indices of rows that form the stable capture region.
    intervals is length N-1 for N frames. start_index refers to frame index.
    We search for first index i where the next `window` intervals are within expected +/- tol*expected.
    Similarly for the end (searching backwards).
    """
    if not intervals:
        return 0, 0
    low = expected * (1 - tol)
    high = expected * (1 + tol)
    n = len(intervals)
    start = 0
    end = n  # intervals index; corresponding to last frame index = end
    # find start
    for i in range(0, n - window + 1):
        window_vals = intervals[i:i+window]
        if all((low <= v <= high) for v in window_vals):
            start = i
            break
    # find end (search backwards)
    for j in range(n - window, -1, -1):
        window_vals = intervals[j:j+window]
        if all((low <= v <= high) for v in window_vals):
            end = j + window  # end is interval index after which frames are stable up to
            break
    # Convert interval indices to frame indices: frames indices are 0..N
    # We want to include the frame at `start` and up to `end` (inclusive)
    frame_start = start
    frame_end = end  # inclusive index into frames; safe clamp
    return frame_start, frame_end


def format_seconds(s: float) -> str:
    if s >= 1.0:
        return f"{s:.3f}s"
    ms = s * 1000.0
    return f"{ms:.1f}ms"


def analyze(path: str, trim: bool = True, tol: float = 0.5, window: int = 5):
    meta, rows = parse_log(path)
    if not rows:
        print("No data rows found in log.")
        return 1
    fps = meta.get("fps", None)
    if fps is None:
        print("Warning: fps not found in header, defaulting to 30")
        fps = 30
    expected = 1.0 / fps

    host_ts = [r[0].timestamp() for r in rows]
    dev_ts = [r[2] for r in rows]
    seqs = [r[1] for r in rows]
    sizes = [r[3] for r in rows]

    host_intervals = diffs_from_timestamps(host_ts)
    dev_intervals = diffs_from_timestamps(dev_ts)

    missing_count, gaps = detect_missing_sequences(seqs)

    total_frames = len(rows)
    duration = host_ts[-1] - host_ts[0] if total_frames > 1 else 0.0
    expected_frames = int(round(duration * fps)) if duration > 0 else total_frames
    measured_fps = total_frames / duration if duration > 0 else float('nan')

    print(f"Log: {path}")
    if "video" in meta:
        print(f"  video: {meta['video']}")
    print(f"  header fps: {fps}  expected interval: {expected:.6f}s")
    print()

    print("Overall:")
    print(f"  Frames (rows): {total_frames}")
    print(f"  Duration (host timestamps): {format_seconds(duration)}")
    print(f"  Expected frames (duration*fps): {expected_frames}")
    print(f"  Measured capture rate: {measured_fps:.3f} fps")
    print(f"  Missing frames (by seq gaps): {missing_count}")
    if gaps:
        print(f"    Gaps (prev->next): {gaps[:5]}{' ...' if len(gaps)>5 else ''}")
    print()

    hi_stats = stats(host_intervals)
    di_stats = stats(dev_intervals)

    def print_stats(name, s):
        if not s:
            print(f"  {name}: no intervals")
            return
        print(f"  {name} intervals: count={s['count']} mean={format_seconds(s['mean'])} median={format_seconds(s['median'])} max={format_seconds(s['max'])} stdev={s['stdev']:.4f}s")

    print("Host timestamp intervals:")
    print_stats("Host", hi_stats)
    print("Device timestamp intervals:")
    print_stats("Device", di_stats)

    # identify extreme waits
    if host_intervals:
        max_wait = max(host_intervals)
        max_idx = host_intervals.index(max_wait)
        print(f"\nMax host wait: {format_seconds(max_wait)} between frame idx {max_idx} -> {max_idx+1} (seq {seqs[max_idx]} -> {seqs[max_idx+1]})")

    # attempt to find stable capture region and optionally trim
    if trim and host_intervals:
        start_idx, end_idx = find_stable_region(host_intervals, expected, tol=tol, window=window)
        # map to frame indices: we include frames from start_idx to end_idx (inclusive)
        # ensure boundaries are safe
        start_idx = max(0, start_idx)
        end_idx = min(len(rows)-1, end_idx)
        trimmed_frames = rows[start_idx:end_idx+1]
        trimmed_count = len(trimmed_frames)
        trimmed_duration = (trimmed_frames[-1][0].timestamp() - trimmed_frames[0][0].timestamp()) if trimmed_count>1 else 0.0
        print(f"\nStable region detection (tol={tol*100:.0f}%, window={window}):")
        print(f"  Suggested stable frames: index {start_idx} .. {end_idx} (count={trimmed_count})")
        print(f"  Stable region duration: {format_seconds(trimmed_duration)}")
        trimmed_expected_frames = int(round(trimmed_duration * fps)) if trimmed_duration>0 else trimmed_count
        trimmed_measured_fps = trimmed_count / trimmed_duration if trimmed_duration>0 else float('nan')
        print(f"  Measured fps in stable region: {trimmed_measured_fps:.3f} fps (expected {fps} fps)")
        # Recompute missing in trimmed region
        trimmed_seqs = [r[1] for r in trimmed_frames]
        trimmed_missing, _ = detect_missing_sequences(trimmed_seqs)
        print(f"  Missing frames inside stable region: {trimmed_missing}")
    print()
    print("Done.")
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_log.py path/to/log.log")
        sys.exit(2)
    path = sys.argv[1]
    tol = 0.5
    window = 5
    # optional args could be added later
    sys.exit(analyze(path, trim=True, tol=tol, window=window))
