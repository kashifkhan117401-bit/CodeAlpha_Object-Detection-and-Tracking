"""
SORT: A Simple, Online and Realtime Tracker
Based on Alex Bewley's SORT (https://github.com/abewley/sort), adapted and
simplified for this project. Uses a Kalman Filter per track + Hungarian
algorithm (via scipy) for detection-to-track association based on IoU.
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou_batch(bb_test, bb_gt):
    """Computes IoU between two sets of boxes in [x1,y1,x2,y2] format."""
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])

    o = wh / (area_test + area_gt - wh + 1e-6)
    return o


def convert_bbox_to_z(bbox):
    """[x1,y1,x2,y2] -> [cx,cy,s,r] (s=scale/area, r=aspect ratio)"""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """[cx,cy,s,r] -> [x1,y1,x2,y2]"""
    w = np.sqrt(max(x[2] * x[3], 0))
    h = x[2] / (w + 1e-6)
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker:
    """Tracks a single object's bounding box using a Kalman Filter."""
    count = 0

    def __init__(self, bbox, cls_id=None, class_name=None):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.cls_id = cls_id
        self.class_name = class_name
        # Vote counts per class seen for this track, e.g. {2: 8, 67: 1}
        # -> used to report a stable majority-vote class instead of
        # whatever the detector said on the most recent single frame.
        self.class_votes = {}
        self._class_names_seen = {}
        if cls_id is not None:
            self.class_votes[cls_id] = 1
            self._class_names_seen[cls_id] = class_name

    def update(self, bbox, cls_id=None, class_name=None):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        if cls_id is not None:
            self.class_votes[cls_id] = self.class_votes.get(cls_id, 0) + 1
            self._class_names_seen[cls_id] = class_name
            # Majority vote: most-seen class wins, not just the latest frame's guess
            best_cls_id = max(self.class_votes, key=self.class_votes.get)
            self.cls_id = best_cls_id
            self.class_name = self._class_names_seen[best_cls_id]

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(row_ind, col_ind)))
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
    unmatched_trackers = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort:
    """Main SORT multi-object tracker."""

    def __init__(self, max_age=15, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5)), cls_ids=None, class_names=None):
        """
        dets: ndarray of [x1,y1,x2,y2,score]
        Returns ndarray of [x1,y1,x2,y2,track_id,cls_id]
        """
        self.frame_count += 1
        cls_ids = cls_ids if cls_ids is not None else [None] * len(dets)
        class_names = class_names if class_names is not None else [None] * len(dets)

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        dets_boxes = dets[:, :4] if len(dets) > 0 else np.empty((0, 4))
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets_boxes, trks[:, :4] if len(trks) > 0 else trks, self.iou_threshold)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4], cls_ids[m[0]], class_names[m[0]])

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4], cls_ids[i], class_names[i])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id], [trk.cls_id if trk.cls_id is not None else -1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 6))
