"""Summarizes capture.py's camera boot log (logs/camera_boot_log.jsonl) -- one
line per connect *attempt*, written by capture.py's connect_device_with_retry
-- to show which camera IDs are actually the ones failing/needing retries.

capture.py randomizes its camera boot order every run (see main()), so a
boot-position bias (e.g. "whichever camera boots last tends to fail") can't
masquerade as a specific-camera bias here -- this script reports both
separately.

Usage:
    python camera_boot_stats.py [path/to/camera_boot_log.jsonl]
"""
import json
import os
import sys
from collections import defaultdict

# Mirrors capture.py's BOOT_LOG_PATH -- duplicated rather than imported so this
# stays a light, dependency-free script usable even where depthai/cv2 (pulled
# in by importing capture.py) aren't installed.
DEFAULT_LOG_PATH = os.path.join("logs", "camera_boot_log.jsonl")


def usb_bus(location):
    """Same "<bus>.<port>"-style convention/caveats as check_usb_speed.py's
    usb_bus() -- devices sharing a bus number are very likely sharing a host
    controller/root hub, so grouping failures by bus (rather than just by
    device_id) can surface a bad port/controller even as different physical
    cameras get plugged into it over time.
    """
    return location.split(".")[0] if "." in location else location


def load_events(path):
    events = []
    if not os.path.exists(path):
        return events
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def summarize(events):
    by_device = defaultdict(lambda: {
        "attempts": 0, "failures": 0, "successes": 0, "sessions": set(),
        "fail_positions": [], "success_positions": [], "usb_locations": set(),
    })
    by_bus = defaultdict(lambda: {"attempts": 0, "failures": 0, "device_ids": set()})
    # (session, device_id) -> did any attempt in that session succeed?
    ever_succeeded = {}

    for e in events:
        device_id = e.get("device_id", "?")
        session = e.get("session", "?")
        pos = e.get("boot_position")
        location = e.get("usb_location")
        d = by_device[device_id]
        d["attempts"] += 1
        d["sessions"].add(session)
        if location:
            d["usb_locations"].add(location)
            b = by_bus[usb_bus(location)]
            b["attempts"] += 1
            b["device_ids"].add(device_id)
        key = (session, device_id)
        if e.get("outcome") == "success":
            d["successes"] += 1
            if pos is not None:
                d["success_positions"].append(pos)
            ever_succeeded[key] = True
        else:
            d["failures"] += 1
            if pos is not None:
                d["fail_positions"].append(pos)
            if location:
                by_bus[usb_bus(location)]["failures"] += 1
            ever_succeeded.setdefault(key, False)

    hard_failures = defaultdict(set)
    for (session, device_id), ok in ever_succeeded.items():
        if not ok:
            hard_failures[device_id].add(session)

    return by_device, hard_failures, by_bus


def build_position_grid(events, key_fn):
    """counts[key][boot_position] = {"success": n, "fail": n} -- how many
    attempts each row-key (a camera, or a USB bus -- see key_fn) has had at
    each boot-order position, split by outcome, so a position-bias (e.g. only
    ever failing near the end of the boot order) can be told apart from "this
    key just hasn't been tested at that position yet" (which would show as
    untested, not merely 0 fails). key_fn(event) -> the row-grouping key, or
    None to skip an event that doesn't have one (e.g. no usb_location logged).
    """
    counts = defaultdict(lambda: defaultdict(lambda: {"success": 0, "fail": 0}))
    positions = set()
    for e in events:
        pos = e.get("boot_position")
        key = key_fn(e)
        if pos is None or key is None:
            continue
        outcome = "success" if e.get("outcome") == "success" else "fail"
        counts[key][pos][outcome] += 1
        positions.add(pos)
    return counts, sorted(positions)


def print_position_grid(title, counts, row_keys, positions, kind, row_label="Device ID"):
    """kind: 'success', 'fail', or 'total'. Cells are blank ('.') where this
    row (a camera or a USB bus) has never been attempted at that boot
    position at all -- distinct from a real 0 (e.g. 0 fails at a position it
    HAS been tried at), so "never tested here" doesn't get misread as
    "always succeeds here".
    """
    col_w = max(4, len(str(len(positions))) + 2)
    print(f"\n{title}")
    print("  " + f"{row_label:<24}" + "".join(f"{p:>{col_w}}" for p in positions))
    for key in row_keys:
        row = counts.get(key, {})
        cells = []
        for p in positions:
            c = row.get(p)
            if c is None:
                cells.append(f"{'.':>{col_w}}")
                continue
            if kind == "success":
                v = c["success"]
            elif kind == "fail":
                v = c["fail"]
            else:
                v = c["success"] + c["fail"]
            cells.append(f"{v:>{col_w}}")
        print("  " + f"{key:<24}" + "".join(cells))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG_PATH
    events = load_events(path)
    if not events:
        print(f"No boot log events found at {path}.")
        return

    by_device, hard_failures, by_bus = summarize(events)

    rows = []
    for device_id, d in by_device.items():
        avg_fail_pos = sum(d["fail_positions"]) / len(d["fail_positions"]) if d["fail_positions"] else None
        # Usually one location; if a device shows more than one, it's been
        # plugged into different ports across runs -- worth knowing when
        # judging whether offences follow the camera or the port.
        locations = ",".join(sorted(d["usb_locations"])) if d["usb_locations"] else "-"
        rows.append((device_id, d["failures"], d["attempts"], len(d["sessions"]),
                     len(hard_failures.get(device_id, ())), avg_fail_pos, locations))
    rows.sort(key=lambda r: r[1], reverse=True)

    print(f"Read {len(events)} boot events from {path}\n")
    print(f"  {'Device ID':<24} {'Offences':<9} {'Attempts':<9} {'Sessions':<9} "
          f"{'Hard fails':<11} {'Avg fail pos':<13} {'USB location(s)':<20}")
    for device_id, failures, attempts, sessions, hard, avg_pos, locations in rows:
        avg_pos_str = f"{avg_pos:.1f}" if avg_pos is not None else "-"
        print(f"  {device_id:<24} {failures:<9} {attempts:<9} {sessions:<9} {hard:<11} "
              f"{avg_pos_str:<13} {locations:<20}")

    position_counts, positions = build_position_grid(events, lambda e: e.get("device_id", "?"))
    if positions:
        device_ids = sorted(by_device.keys())
        print("\nBoot-position coverage by camera (columns are boot_position, "
              "'.' = never attempted there):")
        print_position_grid("Totals (successes + fails):", position_counts, device_ids, positions, "total")
        print_position_grid("Successes:", position_counts, device_ids, positions, "success")
        print_position_grid("Fails:", position_counts, device_ids, positions, "fail")

    bus_position_counts, bus_positions = build_position_grid(
        events, lambda e: usb_bus(e["usb_location"]) if e.get("usb_location") else None,
    )
    if bus_positions:
        bus_keys = sorted(by_bus.keys())
        print("\nBoot-position coverage by USB bus (columns are boot_position, "
              "'.' = never attempted there):")
        print_position_grid("Totals (successes + fails):", bus_position_counts, bus_keys,
                             bus_positions, "total", row_label="USB bus")
        print_position_grid("Successes:", bus_position_counts, bus_keys, bus_positions,
                             "success", row_label="USB bus")
        print_position_grid("Fails:", bus_position_counts, bus_keys, bus_positions,
                             "fail", row_label="USB bus")

    if by_bus:
        print(f"\n  {'USB bus':<10} {'Offences':<9} {'Attempts':<9} {'Camera(s) seen on it':<22}")
        bus_rows = sorted(by_bus.items(), key=lambda kv: kv[1]["failures"], reverse=True)
        for bus, b in bus_rows:
            print(f"  {bus:<10} {b['failures']:<9} {b['attempts']:<9} "
                  f"{', '.join(sorted(b['device_ids'])):<22}")
        multi_camera_buses = [bus for bus, b in bus_rows if len(b["device_ids"]) > 1 and b["failures"] > 0]
        if multi_camera_buses:
            print(f"  Bus(es) with failures AND more than one different camera seen on them over "
                  f"time: {', '.join(multi_camera_buses)} -- points at that port/controller/hub "
                  f"itself rather than any one camera.")

    total_failures = sum(r[1] for r in rows)
    if total_failures == 0:
        print("\nNo failed connect attempts recorded -- nothing to investigate yet.")
        return

    worst = rows[0]
    print(f"\nMost offences: {worst[0]} -- {worst[1]} failed attempt(s) across {worst[3]} "
          f"session(s) seen, {worst[4]} session(s) it never connected in at all.")

    # Position-bias check: if failures cluster at high boot_position values
    # regardless of device, that points at a positional (power/bandwidth)
    # cause rather than one bad camera -- compare each device's average
    # failing position against the overall average failing position.
    all_fail_positions = [p for d in by_device.values() for p in d["fail_positions"]]
    if all_fail_positions:
        overall_avg = sum(all_fail_positions) / len(all_fail_positions)
        print(f"Average boot_position across ALL failed attempts: {overall_avg:.1f} "
              f"(higher = later in the boot order that run).")
        print("If that number stays high regardless of which device shows up in the "
              "table above, it's likely positional (e.g. power/bandwidth from earlier "
              "cameras) rather than a specific bad camera; if a few device IDs "
              "consistently dominate 'Offences' regardless of their avg fail position, "
              "it's likely those specific cameras/cables/ports.")


if __name__ == "__main__":
    main()
