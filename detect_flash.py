"""Flash-based sync detector for multi-camera OAK takes.

Finds a small LED indicator in each camera by per-pixel min/max analysis,
tracks its brightness at full 4K resolution, detects a double-flash pattern
(two ~100ms flashes separated by ~1000ms), and reports per-camera frame
offsets vs cam0.

Usage:
    python detect_flash.py /mnt/recordings/20260925_153809/take_3
    python detect_flash.py /mnt/recordings/20260925_153809/take_3 \\
        --out flash_alignment.png --sigma 3.5
"""

import argparse
import glob
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align_session import parse_timestamp_log, parse_log_header


# ---------------------------------------------------------------------------
# LED location detection
# ---------------------------------------------------------------------------

_DS = 2   # spatial downsample (INTER_NEAREST, 2× preserves 2-px LED)


def find_led_location(mjpeg_path, log_path,
                      ref_host_ts=None,
                      scan_start_s=1.0,
                      scan_end_s=8.0,
                      flash_window_s=2.5):
    """Find the LED pixel using isolated per-frame diff spikes.

    The LED turning on/off creates a 1-frame spike in max(|frame_t - frame_{t-1}|)
    that is 3-4× above the surrounding background level (person movement).
    We find frames whose global max-diff is an outlier vs. their local neighborhood,
    then return the pixel location of the largest such spike.

    Decodes at 2× INTER_NEAREST (preserves the 2-px LED).
    """
    log_entries = parse_timestamp_log(log_path)
    cam_start_host = log_entries[0]['host_ts_s']

    if ref_host_ts is not None:
        search_start_fr = max(0, next(
            (i for i, e in enumerate(log_entries)
             if e['host_ts_s'] >= ref_host_ts - 0.5), 0))
        search_end_fr = next(
            (i for i, e in enumerate(log_entries)
             if e['host_ts_s'] >= ref_host_ts + flash_window_s),
            len(log_entries) - 1)
    else:
        search_start_fr = next(
            (i for i, e in enumerate(log_entries)
             if e['host_ts_s'] - cam_start_host >= scan_start_s), 0)
        search_end_fr = next(
            (i for i, e in enumerate(log_entries)
             if e['host_ts_s'] - cam_start_host >= scan_end_s),
            len(log_entries) - 1)

    cap = cv2.VideoCapture(mjpeg_path)
    prev_gray = None
    # Store per-frame: (max_diff_value, argmax_y, argmax_x)
    frame_records = []
    frame_idx = 0

    while frame_idx <= search_end_fr:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        small = cv2.resize(f, (w // _DS, h // _DS), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.int16)

        if frame_idx >= search_start_fr and prev_gray is not None:
            diff = np.abs(gray - prev_gray)
            mx = int(diff.max())
            my, mx_x = np.unravel_index(diff.argmax(), diff.shape)
            frame_records.append((frame_idx, mx, int(my), int(mx_x)))

        prev_gray = gray
        frame_idx += 1

    cap.release()

    if not frame_records:
        return None

    vals = np.array([r[1] for r in frame_records], dtype=np.float32)

    # Local background: median over ±20 frames
    half = 20
    background = np.array([
        float(np.median(vals[max(0, i-half):i+half+1]))
        for i in range(len(vals))
    ], dtype=np.float32)

    # Isolated spike: global max-diff is >> local median (ratio > 2.5)
    ratio = vals / np.maximum(background, 5.0)

    # Find the frame record with the highest ratio — that's the LED transition
    best_i = int(np.argmax(ratio))
    if ratio[best_i] < 1.5:
        return None   # no isolated spike found

    _, _, best_y_ds, best_x_ds = frame_records[best_i]
    return int(best_x_ds * _DS + _DS // 2), int(best_y_ds * _DS + _DS // 2)


# ---------------------------------------------------------------------------
# LED signal extraction
# ---------------------------------------------------------------------------

def led_signal(mjpeg_path, led_x, led_y, patch_half=8):
    """Track max pixel in a (2*patch_half) × (2*patch_half) patch at full res.

    Returns np.ndarray shape (N,).
    """
    x0, x1 = max(0, led_x - patch_half), led_x + patch_half
    y0, y1 = max(0, led_y - patch_half), led_y + patch_half

    cap = cv2.VideoCapture(mjpeg_path)
    signal = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        patch = f[y0:y1, x0:x1]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        signal.append(float(gray.max()))
    cap.release()
    return np.array(signal, dtype=np.float32)


# ---------------------------------------------------------------------------
# Flash detection
# ---------------------------------------------------------------------------

def find_flash_events(signal, thresh_sigma=3.5):
    """Contiguous above-threshold regions → list of (peak_frame, peak_val).

    Uses a local-minimum reference so that a slowly-moving bright object
    (person walking through) doesn't inflate the baseline.  The threshold is
    local_min_neighbourhood + thresh_sigma * local_spread, where local_min is
    the running minimum over a ±30-frame window and local_spread is the 10th
    percentile of the abs-diff signal (proxy for per-frame LED step size).

    Falls back to a global IQR threshold when the local approach finds nothing.
    """
    sig = signal.astype(np.float64)
    N = len(sig)

    # Local minimum over a ±60-frame sliding window (keeps up with slow drift)
    half = 60
    local_min = np.array([sig[max(0, i-half):i+half+1].min() for i in range(N)],
                         dtype=np.float64)

    # Frame-to-frame absolute differences — the LED creates a sudden large jump
    diffs = np.abs(np.diff(sig, prepend=sig[0]))
    # Use 90th percentile of diffs as the "typical large change" scale
    diff_scale = float(np.percentile(diffs, 90))
    diff_scale = max(diff_scale, 5.0)

    # Threshold: current value must be thresh_sigma×diff_scale above local min
    thresh_arr = local_min + thresh_sigma * diff_scale
    above = sig > thresh_arr

    events = []
    in_event = False
    event_start = 0
    for i, a in enumerate(above):
        if a and not in_event:
            in_event = True
            event_start = i
        elif not a and in_event:
            in_event = False
            region = sig[event_start:i]
            peak_idx = event_start + int(np.argmax(region))
            # Use onset (first above-threshold frame), not peak, for alignment accuracy
            events.append((event_start, float(sig[peak_idx])))
    if in_event:
        region = sig[event_start:]
        peak_idx = event_start + int(np.argmax(region))
        events.append((event_start, float(sig[peak_idx])))

    return events


def find_double_flash(events, min_gap=20, max_gap=45):
    """First pair of events with peak separation in [min_gap, max_gap] frames."""
    for i in range(len(events) - 1):
        for j in range(i + 1, len(events)):
            gap = events[j][0] - events[i][0]
            if min_gap <= gap <= max_gap:
                return events[i][0], events[j][0]
    return None, None


# ---------------------------------------------------------------------------
# Per-take analysis
# ---------------------------------------------------------------------------

def analyze_take(take_dir, sigma=3.5, min_gap=20, max_gap=45, patch_half=3,
                 flash_window_s=2.5):
    """Detect double flash in every camera, compute offsets vs cam0.

    Returns (results_dict, fps).
    """
    mjpeg_paths = sorted(glob.glob(os.path.join(take_dir, "video_cam*.mjpeg")))
    if not mjpeg_paths:
        raise FileNotFoundError(f"No video_cam*.mjpeg in {take_dir}")

    results = {}
    ref_flash1_idx = None
    ref_flash1_device_ts = None
    ref_flash1_host_ts = None   # host_ts_s of cam0 flash1 (used to guide other cams)
    fps = 30.0

    for mjpeg_path in mjpeg_paths:
        cam_label = (os.path.basename(mjpeg_path)
                     .replace("video_", "").replace(".mjpeg", ""))
        log_path = os.path.join(take_dir, f"frame_timestamps_{cam_label}.log")

        log_entries = []
        if os.path.exists(log_path):
            _, _, log_fps, _ = parse_log_header(log_path)
            if log_fps:
                fps = log_fps
            log_entries = parse_timestamp_log(log_path)

        # Find LED location.
        # cam0: scan a restricted early window (flash expected in first ~8s).
        # Other cams: use cam0's flash host_ts as anchor so we search the right window.
        print(f"  {cam_label}: finding LED location...", flush=True)
        led_xy = find_led_location(
            mjpeg_path, log_path,
            ref_host_ts=ref_flash1_host_ts,   # None for cam0
            flash_window_s=flash_window_s,
            scan_start_s=1.0,
            scan_end_s=8.0,
        )
        if led_xy is None:
            print(f"  {cam_label}: LED not found — skipping")
            results[cam_label] = {
                "signal": np.array([]), "events": [], "flash1": None,
                "flash2": None, "led_xy": None,
                "flash1_device_ts": None,
                "offset_frames": None, "offset_ms": None,
                "ts_implied_offset_ms": None, "residual_ms": None,
            }
            continue

        lx, ly = led_xy
        print(f"  {cam_label}: LED at ({lx},{ly}), extracting signal...", flush=True)
        sig = led_signal(mjpeg_path, lx, ly, patch_half=patch_half)

        events = find_flash_events(sig, thresh_sigma=sigma)
        flash1, flash2 = find_double_flash(events, min_gap=min_gap, max_gap=max_gap)

        flash1_device_ts = None
        if flash1 is not None and log_entries and flash1 < len(log_entries):
            flash1_device_ts = log_entries[flash1]['device_ts_s']

        results[cam_label] = {
            "signal": sig,
            "events": events,
            "flash1": flash1,
            "flash2": flash2,
            "led_xy": led_xy,
            "flash1_device_ts": flash1_device_ts,
        }

        if cam_label == "cam0":
            if flash1 is not None:
                ref_flash1_idx = flash1
                ref_flash1_device_ts = flash1_device_ts
                if log_entries and flash1 < len(log_entries):
                    ref_flash1_host_ts = log_entries[flash1]['host_ts_s']

    # Compute offsets vs cam0.
    frame_period_ms = 1000.0 / fps
    for cam_label, r in results.items():
        f1 = r["flash1"]
        offset_frames = (f1 - ref_flash1_idx) if (
            f1 is not None and ref_flash1_idx is not None) else None
        offset_ms = offset_frames * frame_period_ms if offset_frames is not None else None

        ts_implied_ms = None
        if (r["flash1_device_ts"] is not None and ref_flash1_device_ts is not None):
            ts_implied_ms = (r["flash1_device_ts"] - ref_flash1_device_ts) * 1000.0

        residual_ms = (offset_ms - ts_implied_ms
                       if offset_ms is not None and ts_implied_ms is not None
                       else None)

        r.update({
            "offset_frames": offset_frames,
            "offset_ms": offset_ms,
            "ts_implied_offset_ms": ts_implied_ms,
            "residual_ms": residual_ms,
        })

    return results, fps


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_alignment(results, fps, out_path):
    """One subplot per camera, shared x-axis. LED signal + flash markers."""
    cams = sorted(results.keys())
    n = len(cams)
    fig, axes = plt.subplots(n, 1, sharex=True,
                             figsize=(16, 2.5 * n),
                             gridspec_kw={"hspace": 0.05})
    if n == 1:
        axes = [axes]

    frame_period_ms = 1000.0 / fps

    for ax, cam in zip(axes, cams):
        r = results[cam]
        sig = r["signal"]
        if len(sig) == 0:
            ax.set_ylabel(f"{cam}  (no signal)", rotation=0, ha="right",
                          va="center", fontsize=9)
            continue

        frames = np.arange(len(sig))
        lo, hi = sig.min(), sig.max()
        norm = (sig - lo) / max(hi - lo, 1.0)

        ax.fill_between(frames, norm, alpha=0.2, color="steelblue")
        ax.plot(frames, norm, lw=0.6, color="steelblue")
        ax.set_ylim(-0.05, 1.15)
        ax.set_yticks([])

        if r["flash1"] is not None:
            ax.axvline(r["flash1"], color="red", lw=2.0, label="flash 1")
        if r["flash2"] is not None:
            ax.axvline(r["flash2"], color="darkorange", lw=2.0, label="flash 2")
        for ev_idx, _ in r["events"]:
            if ev_idx not in (r["flash1"], r["flash2"]):
                ax.axvline(ev_idx, color="gray", lw=0.8, ls="--", alpha=0.4)

        led = r.get("led_xy")
        led_str = f"  LED=({led[0]},{led[1]})" if led else ""

        off_f = r["offset_frames"]
        off_ms = r["offset_ms"]
        res_ms = r["residual_ms"]
        if off_f is not None:
            sign = "+" if off_f >= 0 else ""
            label = (f"{cam}{led_str}   offset {sign}{off_f} fr ({sign}{off_ms:.1f} ms)"
                     + (f"   residual {'+' if res_ms >= 0 else ''}{res_ms:.1f} ms"
                        if res_ms is not None else ""))
        else:
            label = f"{cam}{led_str}   flash not detected"

        ax.set_ylabel(label, rotation=0, ha="right", va="center",
                      fontsize=9, labelpad=6)

        if cam == cams[0] and r["flash1"] is not None:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

    axes[-1].set_xlabel("Frame index", fontsize=10)
    fig.suptitle(
        f"Flash alignment — {os.path.basename(take_dir_for_title)}"
        f" | fps={fps:.0f}  frame={frame_period_ms:.1f} ms",
        fontsize=11, y=1.002,
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_report(results, fps):
    frame_period_ms = 1000.0 / fps
    hdr = (f"{'Camera':<8} {'flash1':>7} {'flash2':>7} "
           f"{'offset_fr':>10} {'offset_ms':>10} "
           f"{'ts_impl_ms':>11} {'residual_ms':>12}")
    print()
    print(hdr)
    print("-" * len(hdr))
    for cam in sorted(results.keys()):
        r = results[cam]

        def _fmt(v, fmt="+.1f"):
            return f"{v:{fmt}}" if v is not None else "—"

        f1 = r["flash1"]
        f2 = r["flash2"]
        print(f"{cam:<8} {str(f1) if f1 is not None else '—':>7} "
              f"{str(f2) if f2 is not None else '—':>7} "
              f"{_fmt(r['offset_frames'], '+d') if r['offset_frames'] is not None else '—':>10} "
              f"{_fmt(r['offset_ms']):>10} "
              f"{_fmt(r['ts_implied_offset_ms']):>11} "
              f"{_fmt(r['residual_ms']):>12}")
    print()
    print(f"Frame period: {frame_period_ms:.2f} ms")
    print(f"residual_ms = pixel_offset_ms - ts_implied_offset_ms")
    print(f"residual > {frame_period_ms:.1f} ms → timestamp alignment has a real error")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

take_dir_for_title = "unknown"   # set in main() for plot title


def main():
    global take_dir_for_title
    parser = argparse.ArgumentParser(
        description="Detect double LED flash and measure inter-camera alignment")
    parser.add_argument("take_dir", nargs="?",
                        default="/mnt/recordings/20260925_153809/take_3")
    parser.add_argument("--out", default=None)
    parser.add_argument("--sigma", type=float, default=3.5)
    parser.add_argument("--min-gap", type=int, default=20)
    parser.add_argument("--max-gap", type=int, default=45)
    parser.add_argument("--patch-half", type=int, default=3,
                        help="Half-size of LED tracking patch in px (default 3 → 6×6)")
    parser.add_argument("--flash-window", type=float, default=2.5,
                        help="Seconds after expected flash to search (default 2.5)")
    args = parser.parse_args()

    take_dir = args.take_dir
    take_dir_for_title = take_dir
    out_path = args.out or "flash_alignment.png"

    print(f"Analyzing: {take_dir}")
    print(f"  sigma={args.sigma}  gap=[{args.min_gap},{args.max_gap}]fr  "
          f"patch_half={args.patch_half}  flash_window={args.flash_window}s")

    results, fps = analyze_take(
        take_dir,
        sigma=args.sigma,
        min_gap=args.min_gap,
        max_gap=args.max_gap,
        patch_half=args.patch_half,
        flash_window_s=args.flash_window,
    )

    print_report(results, fps)
    plot_alignment(results, fps, out_path)


if __name__ == "__main__":
    main()
