import cv2
import numpy as np
from pathlib import Path


class EdgeSnapSegmentation:
    """Paint rough edges -> auto-fill inside -> snap to exact boundaries."""
    
    def __init__(self, frame, window_name="Paint Rough Edges"):
        self.frame = frame.copy()
        self.original_frame = frame.copy()
        self.display_frame = frame.copy()
        self.window_name = window_name
        
        # Edge painting
        self.edge_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        self.filled_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        self.final_mask = None
        self.segmented_points = None
        
        # Pre-compute edge map for snapping
        self.edge_map = self._compute_edge_map()
        
        # Drawing state
        self.brush_size = 15
        self.is_painting = False
        self.last_point = None
        
        # Processing state
        self.auto_filled = False
        self.edges_snapped = False
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
    
    def _compute_edge_map(self):
        """Compute edge strength map for snapping."""
        # Bilateral filter to preserve strong edges
        filtered = cv2.bilateralFilter(self.frame, 9, 75, 75)
        
        # Convert to grayscale
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        
        # Compute edges using Canny
        edges = cv2.Canny(gray, 30, 100)
        
        # Dilate slightly to make edges more catchable
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        return edges
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse for edge painting."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_painting = True
            self.last_point = (x, y)
            cv2.circle(self.edge_mask, (x, y), self.brush_size, 255, -1)
            self._update_display()
        
        elif event == cv2.EVENT_MOUSEMOVE and self.is_painting:
            # Draw line from last point to current
            if self.last_point:
                cv2.line(self.edge_mask, self.last_point, (x, y), 255, self.brush_size * 2)
            cv2.circle(self.edge_mask, (x, y), self.brush_size, 255, -1)
            self.last_point = (x, y)
            self._update_display()
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_painting = False
            self.last_point = None
    
    def _auto_fill_interior(self):
        """Automatically fill the interior of painted edges."""
        print("Auto-filling interior...")
        
        # Close any gaps in the painted edges
        kernel = np.ones((15, 15), np.uint8)
        closed_edges = cv2.morphologyEx(self.edge_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("⚠ No closed region found. Paint a complete boundary.")
            return False
        
        # Fill the largest contour
        largest = max(contours, key=cv2.contourArea)
        self.filled_mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(self.filled_mask, [largest], -1, 255, -1)
        
        self.auto_filled = True
        print("✓ Interior filled")
        return True
    
    def _snap_to_edges(self):
        """Snap the filled region to nearby strong edges - conservative approach."""
        print("Snapping to precise edges...")
        
        if self.filled_mask is None or np.sum(self.filled_mask) == 0:
            print("⚠ Fill the region first (press 'f')")
            return False
        
        # Start with the filled region and only expand to strong edges
        snapped_mask = self.filled_mask.copy()
        
        # Convert to LAB for better color analysis
        lab = cv2.cvtColor(self.frame, cv2.COLOR_BGR2LAB)
        
        # Get very tight statistics from the center of filled region
        # Erode to get core region
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        core_region = cv2.erode(self.filled_mask, kernel_erode)
        
        if np.sum(core_region) > 100:
            region_pixels = lab[core_region == 255]
            mean_color = np.mean(region_pixels, axis=0)
            std_color = np.std(region_pixels, axis=0)
        else:
            # Fallback to full region
            region_pixels = lab[self.filled_mask == 255]
            mean_color = np.mean(region_pixels, axis=0)
            std_color = np.std(region_pixels, axis=0)
        
        # Very tight color threshold - only similar colors
        threshold = np.mean(std_color) * 2.0
        
        # Compute color distance
        diff = lab.astype(np.float32) - mean_color
        distance = np.sqrt(np.sum(diff**2, axis=2))
        
        # Create tight color mask
        color_mask = (distance < threshold).astype(np.uint8) * 255
        
        # Use edge map to STOP expansion at strong edges
        # Invert edges: high values at edges, low inside regions
        edge_barrier = self.edge_map.astype(np.float32)
        
        # Create distance transform from filled region
        dist_transform = cv2.distanceTransform(self.filled_mask, cv2.DIST_L2, 5)
        
        # Only allow expansion up to 20 pixels from original boundary
        max_expansion = 20
        expansion_zone = (dist_transform <= max_expansion).astype(np.uint8) * 255
        
        # Combine: must be in color mask AND expansion zone AND low edge strength
        edge_threshold = 100  # Don't cross strong edges
        no_strong_edges = (edge_barrier < edge_threshold).astype(np.uint8) * 255
        
        snapped_mask = cv2.bitwise_and(color_mask, expansion_zone)
        snapped_mask = cv2.bitwise_and(snapped_mask, no_strong_edges)
        
        # Ensure we keep the original filled region
        snapped_mask = cv2.bitwise_or(snapped_mask, self.filled_mask)
        
        # Get only the main connected component
        num_labels, labels = cv2.connectedComponents(snapped_mask)
        
        # Find component that overlaps most with original filled region
        overlap_counts = []
        for label_id in range(1, num_labels):
            component = (labels == label_id).astype(np.uint8) * 255
            overlap = np.sum((component > 0) & (self.filled_mask > 0))
            overlap_counts.append((label_id, overlap))
        
        if overlap_counts:
            best_label = max(overlap_counts, key=lambda x: x[1])[0]
            snapped_mask = (labels == best_label).astype(np.uint8) * 255
        
        # Very light morphological smoothing
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        snapped_mask = cv2.morphologyEx(snapped_mask, cv2.MORPH_CLOSE, kernel_smooth, iterations=1)
        
        # Fill only small holes (< 200 pixels)
        contours, hierarchy = cv2.findContours(snapped_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            for i in range(len(contours)):
                if hierarchy[0][i][3] != -1:  # Is a hole
                    area = cv2.contourArea(contours[i])
                    if area < 200:
                        cv2.drawContours(snapped_mask, [contours[i]], -1, 255, -1)
        
        self.final_mask = snapped_mask
        self.edges_snapped = True
        print("✓ Edges snapped conservatively to boundaries")
        return True
    
    def _extract_contour(self):
        """Extract final contour points."""
        mask = self.final_mask if self.edges_snapped else self.filled_mask
        
        if mask is None or np.sum(mask) == 0:
            print("⚠ No mask to extract")
            return False
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Simplify contour
        epsilon = 0.002 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        
        self.segmented_points = [tuple(pt[0]) for pt in approx]
        print(f"✓ Extracted {len(self.segmented_points)} points")
        return True
    
    def _update_display(self):
        """Update the display."""
        self.display_frame = self.original_frame.copy()
        
        # Show painted edges in blue
        if np.sum(self.edge_mask) > 0:
            self.display_frame[self.edge_mask > 0] = [255, 200, 0]  # Cyan
        
        # Show filled region in green (semi-transparent)
        if self.auto_filled and np.sum(self.filled_mask) > 0:
            overlay = self.display_frame.copy()
            overlay[self.filled_mask > 0] = [0, 255, 0]  # Green
            cv2.addWeighted(overlay, 0.4, self.display_frame, 0.6, 0, self.display_frame)
        
        # Show snapped edges in bright green
        if self.edges_snapped and self.final_mask is not None:
            overlay = self.display_frame.copy()
            overlay[self.final_mask > 0] = [0, 255, 0]  # Bright green
            cv2.addWeighted(overlay, 0.5, self.display_frame, 0.5, 0, self.display_frame)
            
            # Draw final contour
            if self.segmented_points:
                pts = np.array(self.segmented_points, dtype=np.int32)
                cv2.polylines(self.display_frame, [pts], True, (0, 255, 255), 3)
        
        # Instructions
        status = "1. Paint edges"
        if self.auto_filled:
            status = "2. Edges filled - press 's' to snap"
        if self.edges_snapped:
            status = "3. Snapped! Press 'ENTER' to save"
        
        cv2.putText(self.display_frame, status, 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.display_frame, "'f' fill | 's' snap | '+/-' brush | 'c' clear | 'q' quit", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(self.display_frame, f"Brush: {self.brush_size}px", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        if self.segmented_points:
            cv2.putText(self.display_frame, f"Points: {len(self.segmented_points)}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow(self.window_name, self.display_frame)
    
    def run(self):
        """Run the segmentation workflow."""
        print("\n" + "=" * 70)
        print("Edge-Snap Segmentation Workflow")
        print("=" * 70)
        print("STEP 1: Paint rough edges around the object (drag mouse)")
        print("STEP 2: Press 'f' to AUTO-FILL the interior")
        print("STEP 3: Press 's' to SNAP edges to precise boundaries")
        print("STEP 4: Press ENTER to SAVE the result")
        print()
        print("Controls:")
        print("  - DRAG mouse to paint edges")
        print("  - 'f' to fill interior")
        print("  - 's' to snap to edges")
        print("  - '+/-' to adjust brush size")
        print("  - 'c' to clear and restart")
        print("  - ENTER to save")
        print("  - 'q' to quit")
        print("=" * 70)
        
        cv2.imshow(self.window_name, self.display_frame)
        
        while True:
            key = cv2.waitKey(50) & 0xFF
            
            if key == ord('f'):
                # Auto-fill interior
                if self._auto_fill_interior():
                    self._update_display()
            
            elif key == ord('s'):
                # Snap to edges
                if self._snap_to_edges():
                    self._extract_contour()
                    self._update_display()
            
            elif key == 13:  # ENTER
                if self.segmented_points and len(self.segmented_points) >= 3:
                    print(f"\n✓ Saving segmentation with {len(self.segmented_points)} points")
                    cv2.destroyAllWindows()
                    return self.segmented_points
                else:
                    print("⚠ Complete the workflow first: paint → fill (f) → snap (s)")
            
            elif key == ord('c'):
                # Clear everything
                self.edge_mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
                self.filled_mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
                self.final_mask = None
                self.segmented_points = None
                self.auto_filled = False
                self.edges_snapped = False
                self.display_frame = self.original_frame.copy()
                cv2.imshow(self.window_name, self.display_frame)
                print("Cleared. Start painting edges again...")
            
            elif key == ord('+') or key == ord('='):
                self.brush_size = min(50, self.brush_size + 5)
                print(f"Brush size: {self.brush_size}")
            
            elif key == ord('-') or key == ord('_'):
                self.brush_size = max(5, self.brush_size - 5)
                print(f"Brush size: {self.brush_size}")
            
            elif key == ord('q'):
                print("Cancelled.")
                cv2.destroyAllWindows()
                return None


def segment_object_from_video(video_path, video_name, shape_name):
    """
    Segment object using edge-snap method.
    
    Args:
        video_path: path to video
        video_name: video identifier  
        shape_name: name for saved shape
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read video")
        return None
    
    print(f"✓ Loaded: {video_path}")
    
    segmenter = EdgeSnapSegmentation(frame)
    points = segmenter.run()
    
    if points:
        try:
            from shapes import SavedShapes
            manager = SavedShapes()
            manager.add_shape(video_name, shape_name, points)
            print(f"\n✓ Successfully saved '{shape_name}' for '{video_name}'")
        except ImportError:
            print(f"\nPoints: {points}")
    
    return points


if __name__ == "__main__":
    print("=== Edge-Snap Segmentation Tool ===\n")
    
    video_path = input("Enter video path: ").strip()
    if not video_path:
        print("No path provided")
        exit()
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error loading video")
        exit()
    
    video_name = Path(video_path).stem
    print(f"✓ Loaded video: {video_name}")
    
    segmenter = EdgeSnapSegmentation(frame)
    points = segmenter.run()
    
    if points:
        shape_name = input("\nEnter shape name: ").strip() or "edge_snapped_object"
        
        try:
            from shapes import SavedShapes
            manager = SavedShapes()
            manager.add_shape(video_name, shape_name, points)
            print(f"\n✓ Successfully saved '{shape_name}' for '{video_name}'")
            print(f"Total points: {len(points)}")
        except ImportError:
            print(f"\nPoints: {points}")