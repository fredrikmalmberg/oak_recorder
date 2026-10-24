# TODOs

## detect_flash.py: measure flash position from the three brightest frames

- Current behaviour: each flash is located at the first frame that crosses the detection threshold.
- Problem: that frame is not reliably the true first frame of the flash. Example: cam2, second flash (take 20260925_153809/take_3), where the detector reports frame 166. The flash crops grid (artifact M4pHHhTBMQtk7j6FwVi82p) shows it.
- Wanted: for each flash, take the three frames with the highest LED brightness and derive the flash position from those, not from the first frame above threshold.
- Open details (decide when implementing): how to combine the three frames (for example the middle one, or a brightness-weighted centre), and how to handle a flash that is split across a frame boundary.
- After the change: rerun on take_3 and re-check the per-camera offsets (current output: cam2 +1, cam4 -2, cam5 -4 frames).

Partly addressed: the new badge detector (`flash_offsets.py`, postprocess) uses the strongest-onset frame of each flash instead of the first above-threshold frame. `detect_flash.py` (fixed LED) is unchanged, so this item is still open for it.

## Capture: let the RGB cycle of the sync flash be fully recorded

- `app.py:2066` (`_do_start_take`) sends `"S"` to the SyncFlashLED *before* the cameras start recording. The firmware plays an RGB colour cycle, waits ~1 s, then fires a white flash.
- In take_1 (20261019_180836) the first flash (RGB cycle) is already lit at frame 0 on cam5 and cam6, so its start is cut off and it is only a cross-check; `flash_offsets.py` uses the white flash (flash 2, about 33 frames later) as the sync reference.
- Option: send `"S"` after the per-camera `begin_recording` loop (or after a short delay) so the whole RGB cycle is visible. Then flash 1 becomes a second usable sync point and `flash_offsets.py` can cross-check every camera. Left as is for now (the current detector works with the white flash).

## WiLoR: set the crop expansion explicitly (rescale_factor 2.0, not 2.5)

- Raise this when the next WiLoR implementation starts.
- `demo.py` and the Gradio demo use `rescale_factor=2.0`; `ViTDetDataset` defaults to 2.5 (looser crop). Pass 2.0 explicitly to match the demo; check packaged variants (e.g. wilor-mini) for their own crop settings.
- Crop = rescale_factor * box size, expanded to BBOX_SHAPE aspect, larger side resized to 256x256; the ViT sees a 256x192 centre region.
- Unverified: BBOX_SHAPE = 192x256 (3:4). Confirm in `wilor/configs`.
- If so, at 2.0 the crop is 1:1 with the source at roughly a 96 px wide or 128 px tall box (77 / 102 px at 2.5). Below ~60 px the crop is upsampled and accuracy drops; above ~130 px extra pixels don't help.
- Adult hand ~17-19 cm, so ~100 px on the long side is about 2 mm/px. Check the smallest handshapes (fists, foreshortened hands).
- Our hand sizes (Sapiens2 keypoint boxes, 48 frames, 3840x2160): median ~145-240 px, p10 ~87-135 px, max ~460 px.
