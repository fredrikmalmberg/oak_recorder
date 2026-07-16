"""Read access to the video frames stored in tmp/Testdata.h5, as distinct from
hand_multiview.py's calibration loader (which reads the same file's pickled
calibration) and its aligned/camN/*.jpg frame discovery (which is for the
separate cam0-cam3 recording sessions, not this h5 layout).

Frame layout: rec/{cam_id}/frames/{frame_key}/color holds JPEG-compressed bytes.
See h5_explore.ipynb for the reference access pattern this mirrors.
"""
import h5py
import numpy as np
import cv2


def discover_h5_cameras(h5_path):
    with h5py.File(h5_path, 'r') as f:
        return sorted(f['rec'].keys())


def discover_h5_frames(h5_path, cam_id):
    with h5py.File(h5_path, 'r') as f:
        return sorted(f[f'rec/{cam_id}/frames'].keys())


def iter_h5_frames(h5_path, cam_ids, frame_keys):
    """Yields (frame_key, {cam_id: img_bgr}) for each frame_key, decoding on
    demand. Opens the h5 file once for the whole iteration rather than
    per-frame, since re-opening for every one of (cameras x frames) reads adds
    up fast; a caller that needs to run multiple detectors per frame should do
    so inside the loop body on the same decoded image, not by iterating twice.
    """
    with h5py.File(h5_path, 'r') as f:
        rec = f['rec']
        for frame_key in frame_keys:
            frames = {}
            for cam_id in cam_ids:
                try:
                    compressed = rec[f'{cam_id}/frames/{frame_key}/color'][()]
                except KeyError:
                    continue
                img = cv2.imdecode(np.frombuffer(compressed, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    frames[cam_id] = img
            yield frame_key, frames
