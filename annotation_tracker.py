import cv2
import numpy as np


class AnnotationTracker:
    def __init__(self, frame, annotation_points, margin=50):
        """
        annotation_points: list of (x,y) points defining the annotation geometry
        """
        self.annotation_points = np.array(annotation_points, dtype=np.float32)
        self.original_annotation_shape = annotation_points.copy()  # Store original shape
        self.margin = margin

        self.status = "tracking"
        self.confidence = 1.0

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Create ROI around annotation
        self.roi = self._compute_roi(self.annotation_points, frame.shape)
        tmpl = frame[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
        self.template = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)

        # Initialize tracking points
        initial_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray = self._preprocess_gray(initial_gray)
        
        # Track BOTH region features AND edge points
        self.region_points = self._detect_features(initial_gray)
        self.edge_points = self._create_edge_points(annotation_points)
        self.points = np.vstack([self.region_points, self.edge_points]) if len(self.region_points) > 0 else self.edge_points

        self.transform = np.eye(2, 3, dtype=np.float32)
        self.template_update_cooldown = 10
        self.last_template_update = -self.template_update_cooldown
        self.frame_index = 0
        
        # Motion smoothing
        self.prev_transforms = []
        self.max_transform_history = 5
        
        # Kalman filter for prediction
        self._init_kalman(annotation_points)
        
        # Lost frame counter
        self.lost_frames = 0
        self.max_lost_frames = 30
        
        # Edge tracking
        self.num_edge_points = len(self.edge_points)
        self.shape_rigidity = 0.7  # How much to preserve original shape (0-1)

    def _init_kalman(self, annotation_points):
        """Initialize Kalman filter for position prediction"""
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                   [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                  [0, 1, 0, 1],
                                                  [0, 0, 1, 0],
                                                  [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # Initialize with center of annotation
        center = np.mean(annotation_points, axis=0)
        self.kalman.statePre = np.array([[center[0]], [center[1]], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[center[0]], [center[1]], [0], [0]], np.float32)

    # ------------------------------
    # Public update
    # ------------------------------
    
    def update(self, frame):
        self.frame_index += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        preprocessed_gray = self._preprocess_gray(gray)

        if self.status == "lost":
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                # Give up on recovery
                return self.annotation_points, "lost", 0.0
            
            self._attempt_recovery(gray)
            return self.annotation_points, self.status, self.confidence

        # Optical flow tracking
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            preprocessed_gray,
            self.points,
            None,
            winSize=(31, 31),  # Larger window for medical video
            maxLevel=4,        # More pyramid levels
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
            minEigThreshold=0.001
        )

        good_old = self.points[status.flatten() == 1]
        good_new = new_points[status.flatten() == 1]

        print(f"Frame {self.frame_index}: {len(self.points)} → {len(good_new)} points tracked")

        # Relaxed threshold for low-texture regions
        if len(good_new) < 4:
            print(f"  Lost: Only {len(good_new)} points")
            self._mark_lost()
            return self.annotation_points, self.status, self.confidence

        # Bidirectional consistency check
        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
            preprocessed_gray,
            self.prev_gray,
            good_new,
            None,
            winSize=(31, 31),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        round_trip_error = np.linalg.norm(
            good_old - back_points, axis=2
        ).flatten()
        
        # More lenient error threshold
        round_trip_mask = (back_status.flatten() == 1) & (round_trip_error <= 2.5)

        good_old = good_old[round_trip_mask]
        good_new = good_new[round_trip_mask]

        if len(good_new) < 4:
            print(f"  Lost: Only {len(good_new)} points after consistency check")
            self._mark_lost()
            return self.annotation_points, self.status, self.confidence

        # Estimate affine transform with more lenient settings
        T, inliers = cv2.estimateAffinePartial2D(  # Partial = similarity transform (better for medical)
            good_old, good_new, 
            method=cv2.RANSAC, 
            ransacReprojThreshold=5.0,  # More lenient
            maxIters=2000,
            confidence=0.95
        )

        if T is None:
            # Fallback 1: Try dense optical flow for fast motion
            print(f"  Affine estimation failed, trying dense optical flow...")
            T = self._estimate_motion_dense(gray, preprocessed_gray)
            
            if T is None:
                # Fallback 2: use median displacement
                print(f"  Dense flow failed, using median displacement")
                displacement = np.median(good_new - good_old, axis=0)
                T = np.array([[1, 0, displacement[0]],
                              [0, 1, displacement[1]]], dtype=np.float32)
                inlier_ratio = 0.5
            else:
                inlier_ratio = 0.6
        else:
            inlier_ratio = np.sum(inliers) / len(good_new) if len(good_new) > 0 else 0

        print(f"  Inlier ratio: {inlier_ratio:.2f}")

        # More lenient inlier threshold
        if inlier_ratio < 0.3:
            print(f"  Lost: Low inlier ratio {inlier_ratio:.2f}")
            self._mark_lost()
            return self.annotation_points, self.status, self.confidence

        # Smooth the transform using history
        T = self._smooth_transform(T)

        # Apply transform to annotation geometry WITH shape preservation
        new_annotation = self._apply_transform(self.annotation_points, T)
        
        # Preserve original shape by constraining deformation
        new_annotation = self._preserve_shape(new_annotation, T)
        self.annotation_points = new_annotation
        
        # Update Kalman filter
        center = np.mean(self.annotation_points, axis=0)
        self.kalman.correct(np.array([[center[0]], [center[1]]], np.float32))

        # Update ROI and re-detect features
        self.roi = self._compute_roi(self.annotation_points, frame.shape)
        
        # Check if ROI is valid and in frame
        if not self._is_roi_valid(frame.shape):
            print(f"  Lost: ROI out of bounds")
            self._mark_lost()
            return self.annotation_points, self.status, self.confidence
        
        # Re-detect features AND edge points
        self.region_points = self._detect_features(gray)
        self.edge_points = self._create_edge_points(self.annotation_points)
        self.points = np.vstack([self.region_points, self.edge_points]) if len(self.region_points) > 0 else self.edge_points
        self.num_edge_points = len(self.edge_points)

        # If we lost too many features, mark as lost
        if len(self.points) < 10:
            print(f"  Lost: Only {len(self.points)} features detected")
            self._mark_lost()
            return self.annotation_points, self.status, self.confidence

        self.prev_gray = preprocessed_gray.copy()
        self.transform = T
        self.confidence = float(inlier_ratio)
        self.status = "tracking"
        self.lost_frames = 0

        # Update template periodically when tracking is good
        if (
            inlier_ratio > 0.75
            and (self.frame_index - self.last_template_update) >= self.template_update_cooldown
        ):
            x1, y1, x2, y2 = self.roi
            if x2 > x1 and y2 > y1:
                roi_gray = gray[y1:y2, x1:x2]
                if roi_gray.size > 0:
                    self.template = roi_gray.copy()
                    self.last_template_update = self.frame_index
                    print(f"  Template updated")

        return self.annotation_points, self.status, self.confidence

    # ------------------------------
    # Shape preservation
    # ------------------------------
    def _create_edge_points(self, annotation_points, points_per_edge=5):
        """Create dense points along the edges of the annotation"""
        # Ensure it's a numpy array
        annotation_points = np.array(annotation_points, dtype=np.float32)
        
        edge_points = []
        n = len(annotation_points)
        
        for i in range(n):
            p1 = annotation_points[i]
            p2 = annotation_points[(i + 1) % n]
            
            # Interpolate points along this edge
            for j in range(points_per_edge):
                t = j / points_per_edge
                point = p1 * (1 - t) + p2 * t
                edge_points.append(point)
        
        return np.array(edge_points, dtype=np.float32).reshape(-1, 1, 2)
    
    def _preserve_shape(self, new_annotation, transform):
        """Preserve the original shape while allowing rigid transformations"""
        # Ensure arrays
        new_annotation = np.array(new_annotation, dtype=np.float32)
        
        # Get the center of the new annotation
        new_center = np.mean(new_annotation, axis=0)
        
        # Get the center of the original shape
        original_center = np.mean(self.original_annotation_shape, axis=0)
        
        # Extract rotation and scale from transform
        # Transform is [[a, b, tx], [c, d, ty]]
        a, b = transform[0, 0], transform[0, 1]
        c, d = transform[1, 0], transform[1, 1]
        
        # Calculate scale and rotation
        scale_x = np.sqrt(a**2 + c**2)
        scale_y = np.sqrt(b**2 + d**2)
        scale = (scale_x + scale_y) / 2
        
        # Limit extreme scaling
        scale = np.clip(scale, 0.5, 2.0)
        
        # Calculate rotation angle
        angle = np.arctan2(c, a)
        
        # Reconstruct the shape: rotate and scale original shape, then translate
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Create rotation matrix
        rotation_matrix = np.array([[cos_a, -sin_a],
                                    [sin_a, cos_a]])
        
        # Apply to original shape (centered at origin)
        original_centered = self.original_annotation_shape - original_center
        rotated_scaled = (rotation_matrix @ original_centered.T).T * scale
        
        # Translate to new center
        preserved_shape = rotated_scaled + new_center
        
        # Blend between tracked position and preserved shape
        # Higher rigidity = more shape preservation
        final_shape = (self.shape_rigidity * preserved_shape + 
                      (1 - self.shape_rigidity) * new_annotation)
        
        return final_shape.astype(np.float32)
    
    # ------------------------------
    # Dense optical flow fallback
    # ------------------------------
    def _estimate_motion_dense(self, gray, preprocessed_gray):
        """Fallback: use dense optical flow for the ROI when sparse flow fails"""
        x1, y1, x2, y2 = self.roi
        
        # Ensure ROI is valid
        if x2 <= x1 or y2 <= y1:
            return None
        
        prev_roi = self.prev_gray[y1:y2, x1:x2]
        curr_roi = preprocessed_gray[y1:y2, x1:x2]
        
        if prev_roi.size == 0 or curr_roi.size == 0:
            return None
        
        # Ensure minimum size
        if prev_roi.shape[0] < 10 or prev_roi.shape[1] < 10:
            return None
        
        try:
            # Compute dense optical flow (Farneback method)
            flow = cv2.calcOpticalFlowFarneback(
                prev_roi, curr_roi, None,
                pyr_scale=0.5,      # Image scale to build pyramids
                levels=5,           # Number of pyramid layers
                winsize=15,         # Averaging window size
                iterations=3,       # Iterations at each pyramid level
                poly_n=5,           # Size of pixel neighborhood
                poly_sigma=1.1,     # Gaussian sigma for polynomial expansion
                flags=0
            )
            
            # Get median flow (more robust than mean)
            flow_reshaped = flow.reshape(-1, 2)
            
            # Filter out extreme outliers
            flow_magnitude = np.linalg.norm(flow_reshaped, axis=1)
            median_mag = np.median(flow_magnitude)
            valid_flow = flow_reshaped[flow_magnitude < median_mag * 3]
            
            if len(valid_flow) < 10:
                return None
            
            median_flow = np.median(valid_flow, axis=0)
            
            print(f"  Dense flow: dx={median_flow[0]:.1f}, dy={median_flow[1]:.1f}")
            
            # Create translation transform from dense flow
            T = np.array([[1, 0, median_flow[0]],
                          [0, 1, median_flow[1]]], dtype=np.float32)
            
            return T
            
        except cv2.error as e:
            print(f"  Dense flow error: {e}")
            return None

    # ------------------------------
    # Feature detection
    # ------------------------------
    def _detect_features(self, gray):
        x1, y1, x2, y2 = self.roi
        roi_gray = gray[y1:y2, x1:x2]
        if roi_gray.size == 0:
            return np.empty((0, 1, 2), dtype=np.float32)

        roi_gray = self._preprocess_gray(roi_gray)

        # Try Shi-Tomasi features first
        points = cv2.goodFeaturesToTrack(
            roi_gray,
            maxCorners=200,  # More features
            qualityLevel=0.005,  # Lower quality threshold for medical images
            minDistance=5,
            blockSize=7,
            useHarrisDetector=False
        )

        # If too few features, try FAST detector
        if points is None or len(points) < 20:
            print(f"  Low features ({len(points) if points is not None else 0}), trying FAST")
            fast = cv2.FastFeatureDetector_create(threshold=5, nonmaxSuppression=True)
            keypoints = fast.detect(roi_gray)
            if keypoints:
                points = np.array([[kp.pt] for kp in keypoints[:200]], dtype=np.float32)

        if points is None or len(points) == 0:
            return np.empty((0, 1, 2), dtype=np.float32)

        # Convert ROI coords → frame coords
        points[:, 0, 0] += x1
        points[:, 0, 1] += y1

        return points.astype(np.float32)

    # ------------------------------
    # Recovery via template matching
    # ------------------------------
    def _attempt_recovery(self, gray):
        h, w = gray.shape
        th, tw = self.template.shape
        
        print(f"  Attempting recovery...")
        
        best_val = 0
        best_loc = None
        best_scale = 1.0
        
        # Try multiple scales
        for scale in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
            scaled_template = cv2.resize(self.template, None, fx=scale, fy=scale)
            sth, stw = scaled_template.shape
            
            if sth >= h - 10 or stw >= w - 10:
                continue
            
            # Use normalized cross-correlation
            res = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_scale = scale
        
        print(f"  Best match: {best_val:.2f} at scale {best_scale:.2f}")
        
        # Lower threshold for recovery
        if best_val > 0.5:
            x, y = best_loc
            tw_scaled = int(tw * best_scale)
            th_scaled = int(th * best_scale)
            
            # Scale annotation points
            old_center = np.mean(self.annotation_points, axis=0)
            new_center = np.array([x + tw_scaled/2, y + th_scaled/2])
            
            # Apply scaling and translation
            offset = self.annotation_points - old_center
            self.annotation_points = new_center + (offset * best_scale)
            
            self.roi = (x, y, x + tw_scaled, y + th_scaled)
            self.points = self._detect_features(gray)
            self.status = "recovered"
            self.confidence = float(best_val)
            self.prev_gray = self._preprocess_gray(gray)
            self.lost_frames = 0
            print(f"  Recovered!")
        else:
            self.confidence *= 0.9

    # ------------------------------
    # Transform smoothing
    # ------------------------------
    def _smooth_transform(self, T):
        """Smooth transform using exponential moving average"""
        self.prev_transforms.append(T)
        if len(self.prev_transforms) > self.max_transform_history:
            self.prev_transforms.pop(0)
        
        if len(self.prev_transforms) == 1:
            return T
        
        # Weighted average (more recent = higher weight)
        weights = np.array([0.5 ** (len(self.prev_transforms) - i - 1) 
                           for i in range(len(self.prev_transforms))])
        weights /= weights.sum()
        
        smoothed = np.zeros_like(T)
        for i, t in enumerate(self.prev_transforms):
            smoothed += weights[i] * t
        
        return smoothed

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
    
    def _is_roi_valid(self, shape):
        """Check if ROI is within frame bounds"""
        h, w = shape[:2]
        x1, y1, x2, y2 = self.roi
        
        # Check if ROI has valid dimensions
        if x2 <= x1 or y2 <= y1:
            return False
        
        # Check if at least 50% of ROI is in frame
        roi_area = (x2 - x1) * (y2 - y1)
        
        x1_clipped = max(0, x1)
        y1_clipped = max(0, y1)
        x2_clipped = min(w, x2)
        y2_clipped = min(h, y2)
        
        visible_area = max(0, x2_clipped - x1_clipped) * max(0, y2_clipped - y1_clipped)
        
        return visible_area >= roi_area * 0.5

    def _apply_transform(self, pts, T):
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
        new_pts = (T @ pts_h.T).T
        return new_pts.astype(np.float32)

    def _preprocess_gray(self, gray):
        return self.clahe.apply(gray)

    def _mark_lost(self):
        self.status = "lost"
        self.confidence *= 0.5