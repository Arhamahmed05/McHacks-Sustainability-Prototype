import cv2
import numpy as np
from pathlib import Path
from annotation_tracker import AnnotationTracker
from shapes import get_all_shapes_for_video

video_path = "Dataset/Lapchole/Lapchole2.mp4"
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
if not ret:
    print("Failed to load video")
    exit()

# Extract video name from path
video_name = Path(video_path).stem  # e.g., "Lapchole1"
print(f"Video: {video_name}")

# Load shapes ONLY for this specific video
shapes_dict = get_all_shapes_for_video(video_name)

if len(shapes_dict) == 0:
    print(f"\nNo saved shapes found for video '{video_name}'!")
    print("Please run shapes.py first to draw and save shapes for this video.")
    exit()

# Create a tracker for each shape belonging to this video
trackers = {}
colors = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
]

print(f"\n✓ Found {len(shapes_dict)} shape(s) for '{video_name}':")

for idx, (shape_name, points) in enumerate(shapes_dict.items()):
    color = colors[idx % len(colors)]
    trackers[shape_name] = {
        'tracker': AnnotationTracker(frame, points),
        'color': color,
        'name': shape_name
    }
    print(f"  - {shape_name}: {len(points)} points")

print("\nStarting tracking...")
cv2.namedWindow(f"Tracking - {video_name}")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print(f"\nEnd of video reached. Processed {frame_count} frames")
        break

    frame_count += 1
    
    # Update and draw each tracker
    for shape_name, tracker_data in trackers.items():
        tracker = tracker_data['tracker']
        color = tracker_data['color']
        name = tracker_data['name']
        
        # Update tracker
        pts, status, conf = tracker.update(frame)
        
        # Draw tracked annotation
        pts_int = pts.astype(int)
        cv2.polylines(frame, [pts_int], True, color, 2)
        
        # Draw points
        for i in range(len(pts_int)):
            cv2.circle(frame, tuple(pts_int[i]), 4, color, -1)
        
        # Add label with status
        center = np.mean(pts_int, axis=0).astype(int)
        label = f"{name}: {status} ({conf:.2f})"
        cv2.putText(frame, label, tuple(center), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Display frame counter and video name
    cv2.putText(frame, f"{video_name} - Frame: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow(f"Tracking - {video_name}", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        print("\nStopped by user")
        break
    elif key == ord('r'):  # R to reset video
        print("\nResetting video...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        
        # Reinitialize all trackers
        for shape_name, points in shapes_dict.items():
            trackers[shape_name]['tracker'] = AnnotationTracker(frame, points)
        
        frame_count = 0

cap.release()
cv2.destroyAllWindows()
print("Done!")