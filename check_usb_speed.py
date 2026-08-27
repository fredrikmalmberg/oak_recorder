"""Diagnostic: reports each connected OAK camera's negotiated USB link speed
and USB location (plus a couple of other quick health signals), to help
track down why cameras aren't running at full throughput.

Mirrors capture.py/oak_camera.py's own device-enumeration pattern
(dai.Device.getAllAvailableDevices()) rather than assuming any particular
pipeline is running -- this just connects to each device, asks it directly,
and disconnects. No pipeline/streaming involved.

Usage:
    python check_usb_speed.py
"""
import depthai as dai

# capture.py's full pipeline (3840x2160 NV12 -> hardware MJPEG encoder, see
# capture.py:91-113) needs USB3-class bandwidth per camera -- SUPER is
# ~5 Gbps, SUPER_PLUS ~10 Gbps. HIGH is USB2-class (480 Mbps, *shared* across
# every device on the same hub/controller) and will bottleneck fps/resolution
# well before the camera's own sensor limit, especially with several
# cameras plugged into the same hub.
USB3_SPEEDS = {dai.UsbSpeed.SUPER, dai.UsbSpeed.SUPER_PLUS}


def describe_cameras(device):
    try:
        features = device.getConnectedCameraFeatures()
    except Exception as exc:
        return f"(could not query cameras: {exc})"
    parts = [f"{f.socket.name}={f.sensorName}@{f.width}x{f.height}" for f in features]
    return ", ".join(parts) if parts else "none reported"


def usb_bus(location):
    """depthai/XLink has no direct "which controller" API -- DeviceInfo.name is
    the closest available signal: a "<bus>.<port>" (or similar) USB location
    string from the OS's own enumeration. Devices sharing a bus number are
    very likely sharing the same host controller/root hub, and therefore its
    bandwidth -- not a guarantee (format isn't formally documented and can
    vary by OS), but a useful grouping hint. Falls back to the raw string if
    it doesn't look like "<bus>.<port>".
    """
    return location.split(".")[0] if "." in location else location


def main():
    device_infos = dai.Device.getAllAvailableDevices()
    if not device_infos:
        print("[Error] No OAK devices found. Check USB connections.")
        return

    print(f"Found {len(device_infos)} OAK device(s):\n")
    rows = []
    for info in device_infos:
        location = info.name  # "<bus>.<port>"-style USB location, see usb_bus()
        print(f"Connecting to {info.deviceId} (USB location {location}) ...")
        try:
            device = dai.Device(info)
        except Exception as exc:
            print(f"  [Error] Failed to connect: {exc}")
            rows.append((info.deviceId, location, "CONNECT FAILED", ""))
            continue

        try:
            speed = device.getUsbSpeed()
            cams = describe_cameras(device)
            flag = "" if speed in USB3_SPEEDS else "  <-- USB2-class, will bottleneck throughput"
            print(f"  USB speed : {speed.name}{flag}")
            print(f"  Cameras   : {cams}")
            try:
                print(f"  Chip temp : {device.getChipTemperature().average:.1f} C")
            except Exception:
                pass
            rows.append((info.deviceId, location, speed.name, cams))
        finally:
            device.close()
        print()

    print("Summary:")
    print(f"  {'Device ID':<24} {'USB location':<14} {'USB speed':<12} Cameras")
    for device_id, location, speed_name, cams in rows:
        print(f"  {device_id:<24} {location:<14} {speed_name:<12} {cams}")

    non_usb3 = [r for r in rows if r[2] not in {"SUPER", "SUPER_PLUS"}]
    if non_usb3:
        print(f"\n[Warning] {len(non_usb3)} device(s) not running at USB3 speed -- "
              f"check cables/hub/ports (USB3 cables and ports are usually blue, "
              f"and each camera ideally wants its own controller, not a shared hub).")

    buses = {}
    for device_id, location, speed_name, _cams in rows:
        buses.setdefault(usb_bus(location), []).append((device_id, location, speed_name))
    shared = {bus: devs for bus, devs in buses.items() if len(devs) > 1}
    if shared:
        print(f"\n[Info] Devices sharing a USB bus (possible bandwidth contention):")
        for bus, devs in shared.items():
            listed = ", ".join(f"{d} ({loc}, {spd})" for d, loc, spd in devs)
            print(f"  bus {bus}: {listed}")


if __name__ == "__main__":
    main()
