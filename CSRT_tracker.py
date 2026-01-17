import cv2
import numpy as np



class AnnotationTrackerCSRT:
    def __init__(self, frame, annotation_points, margin=40):
        """
        annotation_points: list of (x,y) points defining the annotation geometry
        """
        self.annotation_points = np.array(annotation_points, dtype=np.float32)
        self.margin = margin

        self.status = "tracking"
        self.confidence = 1.0

        # Create initial ROI
        self.roi = self._compute_roi(self.annotation_points, frame.shape)
        self.bbox = self._roi_to_bbox(self.roi)

        # Create CSRT tracker
        self.tracker = ctracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, self.bbox)

        self.prev_bbox = self.bbox


    # ------------------------------
    # Public update
    # ------------------------------
    def update(self, frame):
        success, bbox = self.tracker.update(frame)

        if not success:
            self._mark_lost()
            return self.annotation_points, self.status, self.confidence

        self.status = "tracking"
        self.confidence = 1.0

        # Compute scale and translation from previous bbox → new bbox
        prev_x, prev_y, prev_w, prev_h = self.prev_bbox
        new_x, new_y, new_w, new_h = bbox

        prev_w = prev_w if prev_w != 0 else 1.0
        prev_h = prev_h if prev_h != 0 else 1.0

        scale_x = new_w / prev_w
        scale_y = new_h / prev_h

        prev_center = np.array([prev_x + prev_w / 2.0, prev_y + prev_h / 2.0], dtype=np.float32)
        new_center = np.array([new_x + new_w / 2.0, new_y + new_h / 2.0], dtype=np.float32)

        # Apply affine transform (scale around bbox center + translation) to annotation geometry
        self.annotation_points = (self.annotation_points - prev_center) * np.array([scale_x, scale_y]) + new_center
        # If rotation is needed, estimate it from tracker output or add a separate feature-based refinement step.

        # Update ROI + bbox
        self.roi = self._compute_roi(self.annotation_points, frame.shape)
        self.prev_bbox = bbox

        return self.annotation_points, self.status, self.confidence


    # ------------------------------
    # Helpers
    # ------------------------------
    def _compute_roi(self, pts, shape):
        h, w = shape[:2]
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)

        x1 = max(int(x_min - self.margin), 0)
        y1 = max(int(y_min - self.margin), 0)
        x2 = min(int(x_max + self.margin), w - 1)
        y2 = min(int(y_max + self.margin), h - 1)

        return (x1, y1, x2, y2)

    def _roi_to_bbox(self, roi):
        x1, y1, x2, y2 = roi
        return (x1, y1, x2 - x1, y2 - y1)

    def _mark_lost(self):
        self.status = "lost"
        self.confidence = 0.0
