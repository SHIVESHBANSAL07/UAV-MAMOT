# import numpy as np
# from scipy.optimize import linear_sum_assignment
# from utils import correct_bbox

# class KalmanTracker:
#     count = 0
#     def __init__(self, bbox):
#         KalmanTracker.count += 1
#         self.id     = KalmanTracker.count
#         self.hits   = 1
#         self.no_det = 0
#         x1, y1, x2, y2 = bbox
#         cx, cy = (x1+x2)/2, (y1+y2)/2
#         w,  h  = x2-x1, y2-y1
#         self.state = np.array([cx, cy, w, h, 0, 0], dtype=float)

#     def predict(self):
#         self.state[0] += self.state[4]
#         self.state[1] += self.state[5]
#         self.no_det   += 1
#         return self.get_bbox()

#     def update(self, bbox):
#         x1, y1, x2, y2 = bbox
#         cx, cy = (x1+x2)/2, (y1+y2)/2
#         w,  h  = x2-x1, y2-y1
#         alpha  = 0.6
#         self.state[4] = alpha*(cx-self.state[0]) + (1-alpha)*self.state[4]
#         self.state[5] = alpha*(cy-self.state[1]) + (1-alpha)*self.state[5]
#         self.state[0], self.state[1] = cx, cy
#         self.state[2], self.state[3] = w,  h
#         self.hits   += 1
#         self.no_det  = 0

#     def get_bbox(self):
#         cx, cy, w, h = self.state[:4]
#         return (cx-w/2, cy-h/2, cx+w/2, cy+h/2)

# def iou(b1, b2):
#     x1 = max(b1[0],b2[0]); y1 = max(b1[1],b2[1])
#     x2 = min(b1[2],b2[2]); y2 = min(b1[3],b2[3])
#     inter = max(0,x2-x1)*max(0,y2-y1)
#     a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
#     a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
#     return inter/(a1+a2-inter+1e-9)

# class MAMOTTracker:
#     def __init__(self, max_age=10, min_hits=2, iou_thresh=0.3):
#         self.trackers   = []
#         self.max_age    = max_age
#         self.min_hits   = min_hits
#         self.iou_thresh = iou_thresh
#         KalmanTracker.count = 0

#     def update(self, detections, dx=0, dy=0):
#         predicted  = [t.predict() for t in self.trackers]
#         corrected  = [correct_bbox(d, dx, dy) for d in detections]
#         if self.trackers and corrected:
#             iou_matrix = np.array([[iou(p,c) for c in corrected]
#                                               for p in predicted])
#             row_ind, col_ind = linear_sum_assignment(-iou_matrix)
#             matched_t, matched_d = set(), set()
#             for r, c in zip(row_ind, col_ind):
#                 if iou_matrix[r,c] >= self.iou_thresh:
#                     self.trackers[r].update(corrected[c])
#                     matched_t.add(r); matched_d.add(c)
#             for i, det in enumerate(corrected):
#                 if i not in matched_d:
#                     self.trackers.append(KalmanTracker(det))
#         else:
#             for det in corrected:
#                 self.trackers.append(KalmanTracker(det))
#         self.trackers = [t for t in self.trackers
#                          if t.no_det <= self.max_age]
#         return [(t.id, t.get_bbox()) for t in self.trackers
#                 if t.hits >= self.min_hits]

import numpy as np
from scipy.optimize import linear_sum_assignment
from utils import correct_bbox

class KalmanTracker:
    count = 0

    def __init__(self, bbox):
        KalmanTracker.count += 1
        self.id = KalmanTracker.count
        self.hits = 1
        self.no_det = 0
        x1, y1, x2, y2 = bbox
        cx, cy = (x1+x2)/2, (y1+y2)/2
        w, h = x2-x1, y2-y1
        self.state = np.array([cx, cy, w, h, 0, 0], dtype=float)

    def predict(self):
        self.state[0] += self.state[4]
        self.state[1] += self.state[5]
        self.no_det += 1
        return self.get_bbox()

    def update(self, bbox):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1+x2)/2, (y1+y2)/2
        w, h = x2-x1, y2-y1
        alpha = 0.6
        self.state[4] = alpha*(cx-self.state[0]) + (1-alpha)*self.state[4]
        self.state[5] = alpha*(cy-self.state[1]) + (1-alpha)*self.state[5]
        self.state[0], self.state[1] = cx, cy
        self.state[2], self.state[3] = w, h
        self.hits += 1
        self.no_det = 0

    def get_bbox(self):
        cx, cy, w, h = self.state[:4]
        return (cx-w/2, cy-h/2, cx+w/2, cy+h/2)


def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


class MAMOTTracker:
    def __init__(self, max_age=10, min_hits=1, iou_thresh=0.3):
        self.trackers = []
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thresh = iou_thresh
        KalmanTracker.count = 0

    def update(self, detections, dx=0, dy=0):
        predicted = [t.predict() for t in self.trackers]
        corrected = [correct_bbox(d, dx, dy) for d in detections]

        if self.trackers and corrected:
            iou_matrix = np.array([[iou(p, c)
                                    for c in corrected]
                                   for p in predicted])
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_t, matched_d = set(), set()

            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_thresh:
                    self.trackers[r].update(corrected[c])
                    matched_t.add(r)
                    matched_d.add(c)

            for i, det in enumerate(corrected):
                if i not in matched_d:
                    self.trackers.append(KalmanTracker(det))
        else:
            for det in corrected:
                self.trackers.append(KalmanTracker(det))

        self.trackers = [t for t in self.trackers
                         if t.no_det <= self.max_age]

        return [(t.id, t.get_bbox()) for t in self.trackers
                if t.hits >= self.min_hits]