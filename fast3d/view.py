"""Player for fast3d results (fast3d step 6).

Fast3DPlayer(server, npz_path) draws a take's triangulated MediaPipe body and
hands (line skeletons) and face (filled with the 898 triangles of MediaPipe's
canonical face model, mediapipe_face_triangles.npy, or as the FaceMesh
tessellation wireframe) into an existing viser server (the app's own scene, or a
standalone one) under one scene root, with its GUI (timeline, play/pause,
speed, raw/smoothed, part toggles) in one folder (inside gui_parent when
given). It runs its own playback
thread, so it does not block the caller; close() removes everything.

Standalone, with camera frustums:

    python -m fast3d.view <take_dir> [--port 8095]
"""
import argparse
import os
import threading
import time

import numpy as np

TRIS_FACE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mediapipe_face_triangles.npy')
COL = {'body': (230, 230, 230), 'left': (80, 160, 255), 'right': (255, 120, 80), 'face': (120, 230, 120)}


class Fast3DPlayer:
    def __init__(self, server, npz_path, root='/fast3d', folder_label='3D playback (fast3d)', fps=None, gui_parent=None):
        self.server, self.root = server, root
        d = np.load(npz_path)
        self.keys = [str(k) for k in d['frame_keys']]
        self.T = len(self.keys)
        self.edges = {'body': d['edges_body'], 'left': d['edges_hand'], 'right': d['edges_hand'], 'face': d['edges_face']}
        tri = d['tris_face'] if 'tris_face' in d else np.load(TRIS_FACE)
        self.tris_face = np.asarray(tri, np.int32)
        self.data = {kind: {p: d[f'{p}_{kind}'] for p in ('body', 'left', 'right', 'face') if f'{p}_{kind}' in d}
                     for kind in ('smoothed', 'raw')}
        if not self.data['smoothed']:
            raise ValueError(f'{npz_path} has no triangulated parts (2D-only run)')
        g = server.gui
        if gui_parent is not None:
            with gui_parent:
                self.folder = g.add_folder(folder_label)
        else:
            self.folder = g.add_folder(folder_label)
        with self.folder:
            self.play = g.add_button('Play / Pause')
            self.speed = g.add_slider('Playback FPS', min=1, max=60, step=1, initial_value=int(fps or 30))
            self.slider = g.add_slider('Frame', min=0, max=self.T - 1, step=1, initial_value=0)
            self.smoothed = g.add_checkbox('Smoothed (Kalman/RTS)', True)
            self.show = {p: g.add_checkbox(f'Show {p}', True) for p in ('body', 'left', 'right', 'face')}
            self.face_filled = g.add_checkbox('Face filled (else wireframe)', True)
        self._face_mesh = None
        self._playing = False
        self.play.on_click(lambda _: setattr(self, '_playing', not self._playing))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _draw(self, i):
        data = self.data['smoothed' if self.smoothed.value else 'raw']
        for p, X in data.items():
            name = f'{self.root}/{p}'
            x = X[i]
            if p == 'face':
                v = x[:468]
                filled = self.show[p].value and self.face_filled.value and np.isfinite(v).all()
                if filled:
                    if self._face_mesh is None:
                        self._face_mesh = self.server.scene.add_mesh_simple(
                            f'{self.root}/face_mesh', v.astype(np.float32), self.tris_face, color=COL['face'],
                            side='double', flat_shading=False)
                    else:
                        self._face_mesh.vertices = v.astype(np.float32)
                    self._face_mesh.visible = True
                elif self._face_mesh is not None:
                    self._face_mesh.visible = False
                if self.face_filled.value:
                    self.server.scene.add_line_segments(name, np.zeros((1, 2, 3), np.float32), COL[p], visible=False)
                    continue
            e = self.edges[p]
            ok = np.isfinite(x).all(1)
            seg = e[ok[e[:, 0]] & ok[e[:, 1]]] if len(e) else e
            vis = self.show[p].value and len(seg) > 0
            if not vis:
                self.server.scene.add_line_segments(name, np.zeros((1, 2, 3), np.float32), COL[p], visible=False)
                continue
            self.server.scene.add_line_segments(name, x[seg].astype(np.float32), COL[p],
                                                thickness=1.0 if p == 'face' else 3.0, thickness_units='screen')

    def _loop(self):
        last = None
        while not self._stop.is_set():
            if self._playing:
                self.slider.value = (self.slider.value + 1) % self.T
            i = self.slider.value
            state = (i, self.smoothed.value, self.face_filled.value, tuple(c.value for c in self.show.values()))
            if state != last:
                self._draw(i)
                last = state
            time.sleep(1.0 / self.speed.value if self._playing else 0.05)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self.server.scene.remove_by_name(self.root)
        except Exception:
            for p in ('body', 'left', 'right', 'face', 'face_mesh'):
                try:
                    self.server.scene.remove_by_name(f'{self.root}/{p}')
                except Exception:
                    pass
        self.folder.remove()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('take_dir')
    ap.add_argument('--port', type=int, default=8095)
    a = ap.parse_args()
    import viser
    from hand_pose import hand_multiview as hmv
    npz = os.path.join(os.path.abspath(a.take_dir), 'processed', 'fast3d', 'fast3d.npz')
    d = np.load(npz)
    server = viser.ViserServer(port=a.port)
    server.gui.configure_theme(dark_mode=True)
    server.scene.set_up_direction((0.0, -1.0, 0.0))      # calibration frame: OpenCV, +Y down
    if 'cam_R' in d:
        hmv.add_camera_frustums(server, {str(c): (R, t) for c, R, t in zip(d['cams'], d['cam_R'], d['cam_t'])},
                                d['cam_K'][0], *(d['image_size'] if 'image_size' in d else (2016, 1520)))
    Fast3DPlayer(server, npz)
    print(f'fast3d player on port {a.port}', flush=True)
    while True:
        time.sleep(1.0)


if __name__ == '__main__':
    main()
