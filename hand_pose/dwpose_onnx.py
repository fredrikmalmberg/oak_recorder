"""Minimal ONNX inference for DWPose (YOLOX person detector + RTMPose COCO-WholeBody
pose estimator), avoiding the mmcv/mmdet/mmpose dependency chain the reference
controlnet_aux implementation requires. Model weights: yolox_l.onnx + dw-ll_ucoco_384.onnx
from https://huggingface.co/yzd-v/DWPose.
"""
import cv2
import numpy as np
import onnxruntime as ort

# COCO-WholeBody 133-keypoint layout.
LEFT_HAND_SLICE = slice(91, 112)
RIGHT_HAND_SLICE = slice(112, 133)


def _preprocess_yolox(img, input_size=(640, 640)):
    padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    ratio = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)), interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: resized_img.shape[0], : resized_img.shape[1]] = resized_img
    padded_img = np.ascontiguousarray(padded_img.transpose(2, 0, 1), dtype=np.float32)
    return padded_img, ratio


def _yolox_decode_grid(outputs, img_size, strides=(8, 16, 32)):
    grids, expanded_strides = [], []
    for stride in strides:
        hsize, wsize = img_size[0] // stride, img_size[1] // stride
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((*grid.shape[:2], 1), stride))
    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    return outputs


def _nms(boxes, scores, nms_thr):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][ovr <= nms_thr]
    return keep


def _detect_persons(session, img_bgr, input_size=(640, 640), score_thr=0.3, nms_thr=0.45):
    img, ratio = _preprocess_yolox(img_bgr, input_size)
    output = session.run(None, {session.get_inputs()[0].name: img[None]})[0]
    predictions = _yolox_decode_grid(output, input_size)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    boxes_xyxy = np.empty_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    boxes_xyxy /= ratio

    person_scores = scores[:, 0]
    valid = person_scores > score_thr
    if not valid.any():
        return np.zeros((0, 5))
    boxes_xyxy, person_scores = boxes_xyxy[valid], person_scores[valid]
    keep = _nms(boxes_xyxy, person_scores, nms_thr)
    return np.concatenate([boxes_xyxy[keep], person_scores[keep, None]], axis=1)


def _bbox_to_center_scale(bbox, padding=1.25):
    x1, y1, x2, y2 = bbox[:4]
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    scale = np.array([x2 - x1, y2 - y1]) * padding
    return center, scale


def _get_warp_matrix(center, scale, output_size):
    src_w, src_h = scale[:2]
    dst_w, dst_h = output_size[:2]
    src_dir = np.array([0., src_w * -0.5])
    dst_dir = np.array([0., dst_w * -0.5])

    def third_point(a, b):
        direction = a - b
        return b + np.r_[-direction[1], direction[0]]

    src = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + src_dir
    src[2] = third_point(src[0], src[1])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0] = [dst_w * 0.5, dst_h * 0.5]
    dst[1] = dst[0] + dst_dir
    dst[2] = third_point(dst[0], dst[1])
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def _fix_aspect_ratio(scale, aspect_ratio):
    w, h = scale
    if w > h * aspect_ratio:
        return np.array([w, w / aspect_ratio])
    return np.array([h * aspect_ratio, h])


def _get_simcc_maximum(simcc_x, simcc_y):
    n, k, _ = simcc_x.shape
    x_locs = np.argmax(simcc_x.reshape(n * k, -1), axis=1)
    y_locs = np.argmax(simcc_y.reshape(n * k, -1), axis=1)
    max_val_x = np.amax(simcc_x.reshape(n * k, -1), axis=1)
    max_val_y = np.amax(simcc_y.reshape(n * k, -1), axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32).reshape(n, k, 2)
    vals = (0.5 * (max_val_x + max_val_y)).reshape(n, k)
    return locs, vals


_POSE_MEAN = np.array([123.675, 116.28, 103.53])
_POSE_STD = np.array([58.395, 57.12, 57.375])


def _estimate_pose(session, bbox, img_bgr, input_size=(288, 384), simcc_split_ratio=2.0):
    center, scale = _bbox_to_center_scale(bbox)
    scale = _fix_aspect_ratio(scale, aspect_ratio=input_size[0] / input_size[1])
    warp_mat = _get_warp_matrix(center, scale, input_size)
    cropped = cv2.warpAffine(img_bgr, warp_mat, input_size, flags=cv2.INTER_LINEAR)

    img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_rgb = (img_rgb - _POSE_MEAN) / _POSE_STD
    input_data = img_rgb.transpose(2, 0, 1)[None].astype(np.float32)

    simcc_x, simcc_y = session.run(None, {session.get_inputs()[0].name: input_data})
    locs, scores = _get_simcc_maximum(simcc_x, simcc_y)
    keypoints = locs[0] / simcc_split_ratio
    keypoints = keypoints / input_size * scale + center - scale / 2
    return keypoints, scores[0]


class DWposeOnnx:
    def __init__(self, det_model_path, pose_model_path, providers=('CPUExecutionProvider',)):
        self.det_session = ort.InferenceSession(det_model_path, providers=list(providers))
        self.pose_session = ort.InferenceSession(pose_model_path, providers=list(providers))

    def detect(self, img_bgr, det_score_thr=0.3):
        """Returns a list of (keypoints[133,2], scores[133]) for each detected person."""
        bboxes = _detect_persons(self.det_session, img_bgr, score_thr=det_score_thr)
        if len(bboxes) == 0:
            return []
        return [_estimate_pose(self.pose_session, bbox, img_bgr) for bbox in bboxes]

    def detect_single_hand(self, img_bgr, det_score_thr=0.3):
        """Single-person, single-hand convenience wrapper: picks the highest-scoring
        detected person, then whichever of their two hands scores higher on average.
        Returns (keypoints[21,2] or None, mean_confidence, per_keypoint_scores[21] or None).
        The per-keypoint scores were always computed internally (SimCC decode gives
        one per keypoint) but previously discarded -- needed for confidence-weighted
        triangulation, where MediaPipe only ever has one whole-hand score to offer."""
        people = self.detect(img_bgr, det_score_thr=det_score_thr)
        if not people:
            return None, 0.0, None
        keypoints, scores = max(people, key=lambda p: p[1][:17].mean())

        left_conf = scores[LEFT_HAND_SLICE].mean()
        right_conf = scores[RIGHT_HAND_SLICE].mean()
        if left_conf >= right_conf:
            hand_kpts, hand_conf, hand_scores = keypoints[LEFT_HAND_SLICE], left_conf, scores[LEFT_HAND_SLICE]
        else:
            hand_kpts, hand_conf, hand_scores = keypoints[RIGHT_HAND_SLICE], right_conf, scores[RIGHT_HAND_SLICE]
        return hand_kpts, float(hand_conf), hand_scores
