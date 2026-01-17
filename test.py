import cv2
# from annotation_tracker import AnnotationTracker
from CSRT_tracker import AnnotationTrackerCSRT

video_path = "Dataset/Lapchole/Lapchole1.mp4"
# video_path = "Dataset/POCUS/Liver.mp4"
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
if not ret:
    print("Failed to load video")
    exit()

clicked_points = []
tracker = None
tracking_started = False

# -------------------------------
# Mouse callback
# -------------------------------
def mouse_callback(event, x, y, flags, param):
    global clicked_points, tracking_started

    if tracking_started:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Point added: ({x}, {y})")

# -------------------------------
# Setup window
# -------------------------------
cv2.namedWindow("Annotate")
cv2.setMouseCallback("Annotate", mouse_callback)

while True:
    if not tracking_started:
        display = frame.copy()

        # Draw clicked points
        for i in range(len(clicked_points)):
            cv2.circle(display, clicked_points[i], 5, (0,255,0), -1)
            if i > 0:
                cv2.line(display, clicked_points[i-1], clicked_points[i], (0,255,0), 2)

        cv2.putText(display, "Click to add points. Press ENTER to start tracking.",
                    (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        cv2.imshow("Annotate", display)

        key = cv2.waitKey(30) & 0xFF
        if key == 13 and len(clicked_points) >= 2:  # ENTER
            tracker = AnnotationTrackerCSRT(frame, clicked_points)
            tracking_started = True
            print("Tracking started")
        elif key == ord('r'):
            clicked_points.clear()
        elif key == 27:
            break

    else:
        ret, frame = cap.read()
        if not ret:
            break

        pts, status, conf = tracker.update(frame)

        # Draw tracked annotation
        for i in range(len(pts)):
            cv2.circle(frame, tuple(pts[i].astype(int)), 5, (0,255,0), -1)
            if i > 0:
                cv2.line(frame, tuple(pts[i-1].astype(int)), tuple(pts[i].astype(int)), (0,255,0), 2)

        cv2.putText(frame, f"{status} ({conf:.2f})", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.imshow("Annotate", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            print("Reset")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            clicked_points.clear()
            tracker = None
            tracking_started = False
        elif key == 27:
            break

cap.release()
cv2.destroyAllWindows()