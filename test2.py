import cv2
import numpy as np
from CSRT_tracker import AnnotationTrackerCSRT

# ------------------------------
# Globals for mouse input
# ------------------------------
clicked_points = []
tracking_started = False


def mouse_callback(event, x, y, flags, param):
    global clicked_points, tracking_started

    if tracking_started:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Point added: ({x}, {y})")


# ------------------------------
# Video
# ------------------------------
cap = cv2.VideoCapture("Dataset/Lapchole/Lapchole1.mp4")
ok, first_frame = cap.read()
if not ok:
    raise RuntimeError("Failed to read video")

display = first_frame.copy()

# ------------------------------
# Setup annotation window
# ------------------------------
cv2.namedWindow("Annotate")
cv2.setMouseCallback("Annotate", mouse_callback)

print("Click to add annotation points. Press ENTER to start tracking.")

# ------------------------------
# Annotation loop
# ------------------------------
while True:
    vis = display.copy()

    # Draw clicked points
    for p in clicked_points:
        cv2.circle(vis, p, 4, (0, 255, 0), -1)

    # Draw polygon if 2+ points
    if len(clicked_points) >= 2:
        cv2.polylines(vis, [np.array(clicked_points)], False, (0, 255, 0), 2)

    cv2.imshow("Annotate", vis)
    key = cv2.waitKey(1)

    # ENTER → start tracking
    if key == 13 and len(clicked_points) >= 2:
        break

    # ESC → quit
    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        exit()

# ------------------------------
# Initialize tracker
# ------------------------------
tracking_started = True
annotation_points = clicked_points.copy()

tracker = AnnotationTrackerCSRT(first_frame, annotation_points)

h, w = first_frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_output = cv2.VideoWriter("tracked.mp4", fourcc, 30, (w, h))

cv2.destroyWindow("Annotate")

# ------------------------------
# Tracking loop (your optical-flow style loop)
# ------------------------------
while True:
    ok, frame = cap.read()
    if not ok:
        print("[ERROR] reached end of file")
        break

    points, status, confidence = tracker.update(frame)

    # Draw annotation geometry
    pts = points.astype(np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw bbox
    x, y, bw, bh = tracker.prev_bbox
    cv2.rectangle(frame, (int(x), int(y)), (int(x+bw), int(y+bh)), (255, 0, 0), 2)

    label = f"{status}  conf={confidence:.2f}"
    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    video_output.write(frame)
    cv2.imshow("CSRT Annotation Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------------------
# Cleanup
# ------------------------------
cap.release()
video_output.release()
cv2.destroyAllWindows()
