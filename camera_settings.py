"""Single source of truth for OAK camera exposure/white-balance defaults.

capture.py established these values (bright, flicker-free studio lighting
assumed -- see its own comments); calibrate.py and oak_camera.py both fall
back to them unless overridden by their own CLI args / config file, so all
three scripts agree on "the same camera" unless a script deliberately asks
for something different.

Resolution and FPS are deliberately NOT included here -- they vary by
script's purpose (capture.py/calibrate.py record at full 4K, oak_camera.py
uses a much lighter preview resolution; calibrate.py also intentionally
runs at a lower FPS than capture.py since it only needs periodic board
detections, not continuous motion).
"""

DEFAULT_CAMERA_SETTINGS = {
    "iso": 200,
    "shutter_us": 4000,
    "wb_k": 4500,
}
