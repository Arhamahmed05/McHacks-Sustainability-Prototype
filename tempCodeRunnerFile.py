import cv2
import numpy as np
import json
from pathlib import Path


class ShapeDrawer:
    """Interactive tool to draw and save annotation shapes."""
    
    def __init__(self, frame, window_name="Draw Shape - Click points, press 's' to save, 'c' to clear, 'q' to quit"):
        self.frame = frame.copy()
        self.display_frame = frame.copy()
        self.points = []
        self.window_name = window_name
        self.drawing = True
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
    
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self._update_display()
    
    def _update_display(self):
        self.display_frame = self.frame.copy()
        
        # Draw all points
        for i, pt in enumerate(self.points):
            cv2.circle(self.display_frame, pt, 5, (0, 255, 0), -1)
            cv2.putText(self.display_frame, str(i+1), (pt[0]+10, pt[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw lines between points
        if len(self.points) > 1:
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(self.display_frame, [pts], False, (0, 255, 0), 2)
        
        # Close the shape visually if we have 3+ points
        if len(self.points) >= 3:
            cv2.line(self.display_frame, self.points[-1], self.points[0], 
                    (0, 255, 0), 2)
        
        cv2.imshow(self.window_name, self.display_frame)
    
    def draw(self):
        """Start interactive drawing session."""
        print("\nInstructions:")
        print("  - Click to add points")
        print("  - Press 's' to SAVE shape")
        print("  - Press 'c' to CLEAR and restart")
        print("  - Press 'q' to QUIT without saving")
        
        cv2.imshow(self.window_name, self.display_frame)
        
        while self.drawing:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                if len(self.points) >= 3:
                    print(f"✓ Shape saved with {len(self.points)} points")
                    self.drawing = False
                else:
                    print("⚠ Need at least 3 points to save a shape")
            
            elif key == ord('c'):
                self.points = []
                self.display_frame = self.frame.copy()
                cv2.imshow(self.window_name, self.display_frame)
                print("Cleared. Start drawing again...")
            
            elif key == ord('q'):
                print("Cancelled.")
                self.points = []
                self.drawing = False
        
        cv2.destroyAllWindows()
        return self.points


class SavedShapes:
    """Manager for saving and loading drawn shapes."""
    
    def __init__(self, shapes_file="saved_shapes.json"):
        self.shapes_file = shapes_file
        self.config_file = "shapes_config.json"
        self.shapes = self._load_shapes()
        self.config = self._load_config()
    
    def _load_shapes(self):
        """Load saved shapes from JSON file."""
        if Path(self.shapes_file).exists():
            with open(self.shapes_file, 'r') as f:
                shapes = json.load(f)
            print(f"✓ Loaded shapes data")
            return shapes
        return {}
    
    def _load_config(self):
        """Load configuration including default shape."""
        if Path(self.config_file).exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {"default_shape": None}
    
    def _save_config(self):
        """Save configuration."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _save_shapes(self):
        """Save shapes to JSON file."""
        with open(self.shapes_file, 'w') as f:
            json.dump(self.shapes, f, indent=2)
    
    def add_shape(self, video_name, shape_name, points):
        """Add a new shape for a specific video."""
        if video_name not in self.shapes:
            self.shapes[video_name] = {}
        
        self.shapes[video_name][shape_name] = points
        self._save_shapes()
        print(f"✓ Saved shape '{shape_name}' for video '{video_name}' with {len(points)} points")
    
    def set_default_shape(self, name):
        """Set a shape as the default to auto-load."""
        if name not in self.shapes:
            raise KeyError(f"Shape '{name}' not found. Cannot set as default.")
        self.config["default_shape"] = name
        self._save_config()
        print(f"✓ Set '{name}' as default shape")
    
    def get_default_shape(self):
        """Get the default shape name."""
        return self.config.get("default_shape")
    
    def get_shape(self, video_name, shape_name):
        """Get a saved shape for a specific video."""
        if video_name not in self.shapes:
            raise KeyError(f"No shapes found for video '{video_name}'")
        if shape_name not in self.shapes[video_name]:
            available = list(self.shapes[video_name].keys())
            raise KeyError(f"Shape '{shape_name}' not found for video '{video_name}'. Available: {available}")
        return self.shapes[video_name][shape_name]
    
    def get_shapes_for_video(self, video_name):
        """Get all shapes for a specific video."""
        if video_name not in self.shapes:
            return {}
        return self.shapes[video_name]
    
    def list_shapes(self, video_name=None):
        """List all saved shapes, optionally filtered by video."""
        if not self.shapes:
            print("No saved shapes found.")
            return []
        
        if video_name:
            if video_name in self.shapes:
                print(f"\nShapes for video '{video_name}':")
                for name, points in self.shapes[video_name].items():
                    print(f"  - '{name}': {len(points)} points")
                return list(self.shapes[video_name].keys())
            else:
                print(f"No shapes found for video '{video_name}'")
                return []
        else:
            print("\nAll saved shapes:")
            for vid_name, vid_shapes in self.shapes.items():
                print(f"\n  Video: '{vid_name}'")
                for shape_name, points in vid_shapes.items():
                    print(f"    - '{shape_name}': {len(points)} points")
            return list(self.shapes.keys())
    
    def delete_shape(self, video_name, shape_name):
        """Delete a saved shape."""
        if video_name in self.shapes and shape_name in self.shapes[video_name]:
            del self.shapes[video_name][shape_name]
            if not self.shapes[video_name]:  # Remove video entry if no shapes left
                del self.shapes[video_name]
            self._save_shapes()
            print(f"✓ Deleted shape '{shape_name}' from video '{video_name}'")
        else:
            print(f"⚠ Shape '{shape_name}' not found for video '{video_name}'")
    
    def visualize_shape(self, video_name, shape_name, frame):
        """Draw a saved shape on a frame."""
        points = self.get_shape(video_name, shape_name)
        display = frame.copy()
        
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(display, [pts], True, (0, 255, 0), 2)
        
        for i, pt in enumerate(points):
            cv2.circle(display, tuple(pt), 5, (0, 0, 255), -1)
        
        return display


def draw_and_save_shape(frame, video_name, shape_name):
    """Draw a shape interactively and save it for a specific video."""
    drawer = ShapeDrawer(frame)
    points = drawer.draw()
    
    if points:
        manager = SavedShapes()
        manager.add_shape(video_name, shape_name, points)
        return points
    return None


def load_saved_shape(video_name, shape_name):
    """Load a previously saved shape for a specific video."""
    manager = SavedShapes()
    return manager.get_shape(video_name, shape_name)


def get_all_shapes_for_video(video_name):
    """Get all shapes saved for a specific video."""
    manager = SavedShapes()
    return manager.get_shapes_for_video(video_name)


def get_default_shape():
    """Get the default shape automatically."""
    manager = SavedShapes()
    default_name = manager.get_default_shape()
    
    if default_name is None:
        raise ValueError("No default shape set. Use SavedShapes().set_default_shape('name')")
    
    return manager.get_shape(default_name)


# Example integration with AnnotationTracker
def tracking_with_saved_shape(video_path, shape_name="default_shape", draw_new=False):
    """
    Run tracking using a saved shape.
    
    Args:
        video_path: path to video file (or 0 for webcam)
        shape_name: name of the saved shape to use
        draw_new: if True, draw a new shape; if False, use existing saved shape
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read video")
        return
    
    # Get annotation points
    if draw_new:
        print(f"\n=== Drawing new shape '{shape_name}' ===")
        annotation_points = draw_and_save_shape(frame, shape_name)
        if not annotation_points:
            print("No shape drawn. Exiting...")
            cap.release()
            return
    else:
        print(f"\n=== Loading saved shape '{shape_name}' ===")
        try:
            manager = SavedShapes()
            annotation_points = manager.get_shape(shape_name)
            print(f"✓ Loaded shape with {len(annotation_points)} points")
        except KeyError as e:
            print(f"Error: {e}")
            print("\nDraw a new shape? (y/n)")
            if input().lower() == 'y':
                annotation_points = draw_and_save_shape(frame, shape_name)
                if not annotation_points:
                    cap.release()
                    return
            else:
                cap.release()
                return
    
    # Show the shape that will be tracked
    display = frame.copy()
    pts = np.array(annotation_points, dtype=np.int32)
    cv2.polylines(display, [pts], True, (0, 255, 0), 2)
    cv2.putText(display, f"Tracking: {shape_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Shape to Track", display)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    
    # Initialize tracker (assuming you have AnnotationTracker imported)
    # from annotation_tracker import AnnotationTracker
    # tracker = AnnotationTracker(frame, annotation_points)
    
    print("\n✓ Ready to track! (Tracker initialization would go here)")
    print(f"Annotation points: {annotation_points}")
    
    cap.release()


if __name__ == "__main__":
    print("=== Shape Drawing & Saving Demo ===\n")
    
    # Ask for video file path
    video_path = input("Enter video file path (or press Enter for test image): ").strip()
    
    if video_path:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"Error: Could not load video from '{video_path}'")
            exit()
        print(f"✓ Loaded video: {video_path}")
        
        # Extract video name from path
        video_name = Path(video_path).stem  # e.g., "Lapchole1" from "Dataset/Lapchole/Lapchole1.mp4"
    else:
        print("No video specified. Creating test image...")
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        cv2.putText(frame, "Test Image - Draw your shape here", (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.rectangle(frame, (50, 50), (590, 430), (100, 100, 100), 2)
        video_name = "test_video"
    
    # Menu
    print(f"\nWorking with video: '{video_name}'")
    print("\nChoose an option:")
    print("1. Draw NEW shape and save")
    print("2. Load EXISTING shape")
    print("3. List all saved shapes")
    print("4. List shapes for this video")
    
    choice = input("\nEnter choice (1/2/3/4): ").strip()
    
    if choice == "1":
        shape_name = input("Enter shape name: ").strip() or "shape_1"
        points = draw_and_save_shape(frame, video_name, shape_name)
    
    elif choice == "2":
        manager = SavedShapes()
        manager.list_shapes(video_name)
        shape_name = input("\nEnter shape name to load: ").strip()
        try:
            points = manager.get_shape(video_name, shape_name)
            vis = manager.visualize_shape(video_name, shape_name, frame)
            cv2.imshow(f"Loaded: {shape_name}", vis)
            print(f"\nShape points: {points}")
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
        except KeyError as e:
            print(e)
    
    elif choice == "3":
        manager = SavedShapes()
        manager.list_shapes()
    
    elif choice == "4":
        manager = SavedShapes()
        manager.list_shapes(video_name)