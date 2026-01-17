import cv2
import numpy as np


class AnnotationTracker:
    def __init__(self, frame, annotation_points, margin=40):
        """
        annotation_points: list of (x,y) points defining the annotation geometry
        """
        self.annotation_points = np.array(annotation_points, dtype=np.float32)
        self.margin = margin

        self.status = "tracking"
        self.confidence = 1.0

        # Create ROI around annotation
        self.roi = self._compute_roi(self.annotation_points, frame.shape)
        tmpl = frame[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
        self.template = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)

        # Initialize tracking points
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.points = self._detect_features(self.prev_gray)

        self.transform = np.eye(2, 3, dtype=np.float32)

    # ------------------------------
    # Public update
    # ------------------------------
    
    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.status == "lost":
            self._attempt_recovery(gray)
            return self.annotation_points, self.status, self.confidence

        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.points,
            np.empty_like(self.points),   # <-- Pylance-safe
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        good_old = self.points[status.flatten() == 1]
        good_new = new_points[status.flatten() == 1]

        if len(good_new) < 6:
            self._mark_lost()
            return self.annotation_points, self.status, self.confidence

        # Estimate affine transform
        T, inliers = cv2.estimateAffine2D(
            good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=3
        )

        if T is None:
            self._mark_lost()
            return self.annotation_points, self.status, self.confidence

        inlier_ratio = np.sum(inliers) / len(inliers)

        if inlier_ratio < 0.6:
            self._mark_lost()
            return self.annotation_points, self.status, self.confidence

        # Apply transform to annotation geometry
        self.annotation_points = self._apply_transform(self.annotation_points, T)

        # Update ROI and points
        self.roi = self._compute_roi(self.annotation_points, frame.shape)
        self.points = self._detect_features(gray)

        self.prev_gray = gray.copy()
        self.transform = T
        self.confidence = float(inlier_ratio)
        self.status = "tracking"

        return self.annotation_points, self.status, self.confidence

    # ------------------------------
    # Feature detection
    # ------------------------------
    def _detect_features(self, gray):
        x1, y1, x2, y2 = self.roi
        roi_gray = gray[y1:y2, x1:x2]

        points = cv2.goodFeaturesToTrack(
            roi_gray,
            maxCorners=150,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7
        )

        if points is None:
            return np.empty((0,1,2), dtype=np.float32)

        # Convert ROI coords → frame coords
        points[:,0,0] += x1
        points[:,0,1] += y1

        return points.astype(np.float32)

    # ------------------------------
    # Recovery via template matching
    # ------------------------------
    def _attempt_recovery(self, gray):
        h, w = gray.shape
        th, tw = self.template.shape

        res = cv2.matchTemplate(gray, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > 0.6:
            x, y = max_loc
            self.roi = (x, y, x+tw, y+th)
            self.points = self._detect_features(gray)
            self.status = "recovered"
            self.confidence = float(max_val)
            self.prev_gray = gray.copy()
        else:
            self.confidence *= 0.9

    # ------------------------------
    # Helpers
    # ------------------------------
    def _compute_roi(self, pts, shape):
        h, w = shape[:2]
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)

        x1 = max(int(x_min - self.margin), 0)
        y1 = max(int(y_min - self.margin), 0)
        x2 = min(int(x_max + self.margin), w-1)
        y2 = min(int(y_max + self.margin), h-1)

        return (x1, y1, x2, y2)

    def _apply_transform(self, pts, T):
        pts_h = np.hstack([pts, np.ones((pts.shape[0],1))])
        new_pts = (T @ pts_h.T).T
        return new_pts.astype(np.float32)

    def _mark_lost(self):
        self.status = "lost"
        self.confidence *= 0.5
    
