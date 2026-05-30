import depthai as dai
import dearpygui.dearpygui as dpg
import cv2
import numpy as np
import time

PANEL_W = 640
PANEL_H = 400

# ── Device detection ──────────────────────────────────────────────────────────

device_infos = dai.Device.getAllAvailableDevices()
if not device_infos:
    print("[Error] No OAK devices found. Check USB connections.")
    exit()

print(f"\nFound {len(device_infos)} OAK device(s):")
for i, info in enumerate(device_infos):
    print(f"  [{i+1}] ID: {info.deviceId}")


def probe_sockets(info):
    try:
        device  = dai.Device(info)
        sockets = {f.socket for f in device.getConnectedCameraFeatures()}
        device.close()
        time.sleep(1)
        return sockets
    except Exception:
        return {dai.CameraBoardSocket.CAM_A}


device_sockets = [probe_sockets(info) for info in device_infos]

def is_stereo(idx):
    return dai.CameraBoardSocket.CAM_B in device_sockets[idx]

for i in range(len(device_infos)):
    print(f"  Camera {i+1}: {'OAK-D' if is_stereo(i) else 'OAK-1'}")

print("Waiting for devices to settle...")
time.sleep(4)

# ── Camera state ──────────────────────────────────────────────────────────────

cameras        = {}
last_frames    = {i: {} for i in range(len(device_infos))}
crashed_cameras = set()
fps_tracker    = {}


def start_camera(idx):
    if idx in cameras:
        return
    info      = device_infos[idx]
    stereo    = is_stereo(idx)
    rgb_res   = (640, 480) if stereo else (1280, 720)
    print(f"Starting camera {idx+1}...")
    try:
        device   = dai.Device(info)
        pipeline = dai.Pipeline(device)

        cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        out_rgb = cam_rgb.requestOutput(rgb_res, type=dai.ImgFrame.Type.BGR888p, fps=30)
        q_rgb   = out_rgb.createOutputQueue()

        q_depth = None
        if stereo:
            cam_left  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
            sd = pipeline.create(dai.node.StereoDepth)
            sd.initialConfig.setConfidenceThreshold(245)
            sd.initialConfig.setLeftRightCheck(True)
            cam_left.requestOutput((640, 400)).link(sd.left)
            cam_right.requestOutput((640, 400)).link(sd.right)
            q_depth = sd.depth.createOutputQueue()

        pipeline.start()
        cameras[idx] = {"pipeline": pipeline, "q_rgb": q_rgb,
                        "q_depth": q_depth, "has_stereo": stereo}
        print(f"  Camera {idx+1} started ({'OAK-D stereo' if stereo else 'OAK-1 color'})")
    except Exception as e:
        print(f"  [Error] Camera {idx+1} failed: {e}")


def stop_camera(idx):
    crashed_cameras.discard(idx)
    if idx not in cameras:
        return
    try:
        cameras[idx]["pipeline"].stop()
    except Exception:
        pass
    del cameras[idx]
    last_frames[idx] = {}
    fps_tracker.pop(idx, None)
    print(f"Camera {idx+1} stopped.")


def toggle_camera(idx):
    if idx >= len(device_infos):
        return
    if idx in crashed_cameras:
        crashed_cameras.discard(idx)
        stop_camera(idx)
        start_camera(idx)
    elif idx in cameras:
        stop_camera(idx)
    else:
        start_camera(idx)


# ── Image helpers ─────────────────────────────────────────────────────────────

def depth_colorize(raw):
    valid = raw[raw > 0]
    if valid.size == 0:
        return np.zeros((*raw.shape, 3), dtype=np.uint8), 0.0, 0.0
    lo = float(np.percentile(valid, 5))
    hi = float(np.percentile(valid, 95))
    if hi <= lo:
        hi = lo + 1
    norm    = np.clip((raw.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[raw == 0] = 0
    return colored, lo, hi


def to_texture(bgr):
    """BGR uint8 (any size) → flat float32 RGBA at PANEL_W×PANEL_H for DPG."""
    if bgr.shape[:2] != (PANEL_H, PANEL_W):
        bgr = cv2.resize(bgr, (PANEL_W, PANEL_H))
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
    return (rgba.astype(np.float32) * (1.0 / 255.0)).flatten()


def placeholder(text, subtext="", bg=(20, 20, 20)):
    img = np.full((PANEL_H, PANEL_W, 3), bg, dtype=np.uint8)
    cv2.putText(img, text,    (20, PANEL_H // 2 - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 1, cv2.LINE_AA)
    if subtext:
        cv2.putText(img, subtext, (20, PANEL_H // 2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1, cv2.LINE_AA)
    return to_texture(img)

# ── DPG setup ─────────────────────────────────────────────────────────────────

dpg.create_context()

# Button themes
with dpg.theme() as _theme_on:
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button,        (20, 120, 20, 255))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (30, 160, 30, 255))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  (15,  90, 15, 255))

with dpg.theme() as _theme_off:
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button,        (55, 55, 55, 255))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (75, 75, 75, 255))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  (40, 40, 40, 255))

with dpg.theme() as _theme_crash:
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button,        (140,  20,  20, 255))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (170,  30,  30, 255))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  (110,  15,  15, 255))

_btn_themes = {"on": _theme_on, "off": _theme_off, "crash": _theme_crash}

# Textures
_blank = [0.0] * (PANEL_W * PANEL_H * 4)

with dpg.texture_registry():
    for i in range(len(device_infos)):
        dpg.add_raw_texture(PANEL_W, PANEL_H, list(_blank),
                            tag=f"tex_rgb_{i}", format=dpg.mvFormat_Float_rgba)
        if is_stereo(i):
            dpg.add_raw_texture(PANEL_W, PANEL_H, list(_blank),
                                tag=f"tex_depth_{i}", format=dpg.mvFormat_Float_rgba)

# Main window
N = len(device_infos)
vp_w = PANEL_W * N + 16
vp_h = PANEL_H * (2 if any(is_stereo(i) for i in range(N)) else 1) + 56

with dpg.window(tag="win", no_title_bar=True, no_move=True,
                no_resize=True, no_scrollbar=True):
    # Toggle buttons
    with dpg.group(horizontal=True):
        for i in range(N):
            kind = "OAK-D" if is_stereo(i) else "OAK-1"
            dpg.add_button(label=f"Cam {i+1} | {kind} | OFF",
                           tag=f"btn_{i}", width=PANEL_W,
                           callback=lambda s, a, u: toggle_camera(u),
                           user_data=i)
            dpg.bind_item_theme(f"btn_{i}", _theme_off)

    dpg.add_separator()

    # Camera panels
    with dpg.group(horizontal=True):
        for i in range(N):
            with dpg.group():
                dpg.add_image(f"tex_rgb_{i}",   tag=f"img_rgb_{i}",   width=PANEL_W, height=PANEL_H)
                if is_stereo(i):
                    dpg.add_image(f"tex_depth_{i}", tag=f"img_depth_{i}", width=PANEL_W, height=PANEL_H)

# Keyboard shortcuts (1–9)
with dpg.handler_registry():
    for i in range(min(9, N)):
        dpg.add_key_press_handler(
            key=getattr(dpg, f"mvKey_{i+1}"),
            callback=lambda s, a, u: toggle_camera(u),
            user_data=i)

dpg.create_viewport(title="OAK Cameras", width=vp_w, height=vp_h,
                    resizable=True, min_width=PANEL_W, min_height=PANEL_H + 56)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("win", True)

# Initial placeholders
for i in range(N):
    kind = "OAK-D" if is_stereo(i) else "OAK-1"
    dpg.set_value(f"tex_rgb_{i}",
                  placeholder(f"Cam {i+1} | {kind}", f"press {i+1} to enable"))
    if is_stereo(i):
        dpg.set_value(f"tex_depth_{i}", placeholder("Depth"))

# ── Update logic ──────────────────────────────────────────────────────────────

def update_cameras():
    for idx in range(N):
        cam  = cameras.get(idx)
        lf   = last_frames[idx]
        kind = "OAK-D" if is_stereo(idx) else "OAK-1"

        if idx in crashed_cameras:
            dpg.set_value(f"tex_rgb_{idx}",
                          placeholder(f"Cam {idx+1} CRASHED",
                                      f"press {idx+1} to retry", bg=(50, 10, 10)))
            dpg.set_item_label(f"btn_{idx}", f"Cam {idx+1} | {kind} | CRASH")
            dpg.bind_item_theme(f"btn_{idx}", _theme_crash)
            continue

        if cam is None:
            dpg.set_value(f"tex_rgb_{idx}",
                          placeholder(f"Cam {idx+1} | {kind}", f"press {idx+1} to enable"))
            dpg.set_item_label(f"btn_{idx}", f"Cam {idx+1} | {kind} | OFF")
            dpg.bind_item_theme(f"btn_{idx}", _theme_off)
            continue

        # Fetch frames
        try:
            rgb_frame = cam["q_rgb"].tryGet()
            if rgb_frame is not None:
                lf["rgb"] = rgb_frame.getCvFrame()
                now  = time.monotonic()
                prev = fps_tracker.get(idx)
                if prev is not None:
                    inst       = 1.0 / max(now - prev, 1e-6)
                    lf["fps"]  = lf.get("fps", inst) * 0.8 + inst * 0.2
                fps_tracker[idx] = now

            if cam["has_stereo"] and cam["q_depth"] is not None:
                d_frame = cam["q_depth"].tryGet()
                if d_frame is not None:
                    colored, lo, hi  = depth_colorize(d_frame.getCvFrame())
                    lf["depth"]      = colored
                    lf["depth_range"] = (lo, hi)

        except Exception as e:
            import traceback
            print(f"  [Camera {idx+1} error: {e}]")
            traceback.print_exc()
            crashed_cameras.add(idx)
            continue

        # RGB panel
        rgb = lf.get("rgb")
        if rgb is not None:
            fps     = lf.get("fps")
            display = rgb.copy()
            label   = (f"Cam {idx+1} | {kind} | {fps:.1f} fps"
                       if fps else f"Cam {idx+1} | {kind}")
            cv2.putText(display, label, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, label, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
            dpg.set_value(f"tex_rgb_{idx}", to_texture(display))

            fps_str = f"{fps:.1f} fps" if fps else ""
            dpg.set_item_label(f"btn_{idx}", f"Cam {idx+1} | {kind} | ON  {fps_str}")
            dpg.bind_item_theme(f"btn_{idx}", _theme_on)

        # Depth panel
        if cam["has_stereo"]:
            depth = lf.get("depth")
            if depth is not None:
                lo, hi  = lf.get("depth_range", (0.0, 0.0))
                display = depth.copy()
                label   = (f"Depth  {lo/1000:.2f} m – {hi/1000:.2f} m"
                           f"   (blue=near  red=far)")
                cv2.putText(display, label, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(display, label, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                dpg.set_value(f"tex_depth_{idx}", to_texture(display))

# ── Resize handling ───────────────────────────────────────────────────────────

_HEADER_H   = 40   # button row + separator
_ANY_STEREO = any(is_stereo(i) for i in range(N))
_last_size  = (0, 0)

def resize_panels():
    global _last_size
    w = dpg.get_viewport_client_width()
    h = dpg.get_viewport_client_height()
    if (w, h) == _last_size:
        return
    _last_size = (w, h)

    col_w     = max(1, w // N)
    avail_h   = max(1, h - _HEADER_H)
    rgb_h     = avail_h // 2 if _ANY_STEREO else avail_h
    depth_h   = avail_h - rgb_h

    for i in range(N):
        dpg.configure_item(f"btn_{i}",     width=col_w)
        dpg.configure_item(f"img_rgb_{i}", width=col_w, height=rgb_h)
        if is_stereo(i):
            dpg.configure_item(f"img_depth_{i}", width=col_w, height=depth_h)

# ── Run ───────────────────────────────────────────────────────────────────────

start_camera(0)

while dpg.is_dearpygui_running():
    resize_panels()
    update_cameras()
    dpg.render_dearpygui_frame()

for idx in list(cameras.keys()):
    stop_camera(idx)
dpg.destroy_context()
print("All cameras stopped.")
