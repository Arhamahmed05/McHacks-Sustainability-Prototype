from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "threads;1"
from pathlib import Path
from annotation_tracker import AnnotationTracker
from CSRT_tracker import AnnotationTrackerCSRT
from shapes import SavedShapes

from flask import Flask, request, jsonify, Response, send_from_directory

app = Flask(__name__)
CORS(app)

# Global state
class VideoState:
    def __init__(self):
        self.video_capture = None
        self.video_path = None
        self.video_name = None
        self.total_frames = 0
        self.current_frame = 0
        self.first_frame = None
        self.is_playing = False
        self.trackers = {}  # {shape_name: {'tracker': tracker_obj, 'color': (r,g,b)}}
        self.shapes_manager = SavedShapes()
        self.colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]

state = VideoState()
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')



@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'is_playing': state.is_playing,
        'frame_count': state.current_frame,
        'total_frames': state.total_frames,
        'video_loaded': state.video_capture is not None,
        'tracker_count': len(state.trackers)
    })

@app.route('/api/load_video', methods=['POST'])
def load_video():
    data = request.json
    video_path = data.get('video_path')
    
    if not video_path:
        return jsonify({'error': 'No video path provided'}), 400
    
    if not os.path.exists(video_path):
        return jsonify({'error': f'Video file not found: {video_path}'}), 404
    
    # Release previous video if exists
    if state.video_capture:
        state.video_capture.release()
    
    # Load new video
    state.video_capture = cv2.VideoCapture(video_path)
    state.video_path = video_path
    state.video_name = Path(video_path).stem  # e.g., "Lapchole1"
    state.total_frames = int(state.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    state.current_frame = 0
    state.is_playing = False
    state.trackers = {}
    
    # Read first frame
    ret, frame = state.video_capture.read()
    if ret:
        state.first_frame = frame
        _, buffer = cv2.imencode('.jpg', frame)
        first_frame_base64 = base64.b64encode(buffer).decode('utf-8')
    else:
        return jsonify({'error': 'Failed to read video'}), 500
    
    # Reset to beginning
    state.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Get saved shapes for this video
    shapes_dict = state.shapes_manager.get_shapes_for_video(state.video_name)
    shape_names = list(shapes_dict.keys())
    
    return jsonify({
        'video_name': state.video_name,
        'total_frames': state.total_frames,
        'first_frame': first_frame_base64,
        'shapes': shape_names
    })

@app.route('/api/list_videos', methods=['GET'])
def list_videos():
    dataset_path = 'Dataset'
    videos = []
    
    if os.path.exists(dataset_path):
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(root, file)
                    videos.append(video_path)
    
    return jsonify({'videos': videos})

@app.route('/api/save_shape', methods=['POST'])
def save_shape():
    data = request.json
    shape_name = data.get('shape_name')
    points = data.get('points')
    
    if not shape_name or not points:
        return jsonify({'error': 'Shape name and points required'}), 400
    
    if len(points) < 3:
        return jsonify({'error': 'Need at least 3 points'}), 400
    
    if not state.video_name:
        return jsonify({'error': 'No video loaded'}), 400
    
    # Save shape for this specific video
    state.shapes_manager.add_shape(state.video_name, shape_name, points)
    
    # Get updated list of shapes for this video
    shapes_dict = state.shapes_manager.get_shapes_for_video(state.video_name)
    shape_names = list(shapes_dict.keys())
    
    return jsonify({
        'success': True,
        'shapes': shape_names
    })

@app.route('/api/delete_shape', methods=['POST'])
def delete_shape():
    data = request.json
    shape_name = data.get('shape_name')
    
    if not state.video_name:
        return jsonify({'error': 'No video loaded'}), 400
    
    state.shapes_manager.delete_shape(state.video_name, shape_name)
    
    # Get updated list
    shapes_dict = state.shapes_manager.get_shapes_for_video(state.video_name)
    shape_names = list(shapes_dict.keys())
    
    return jsonify({
        'success': True,
        'shapes': shape_names
    })

@app.route('/api/initialize_trackers', methods=['POST'])
def initialize_trackers():
    data = request.json
    shape_names = data.get('shapes', [])
    tracker_type = data.get('tracker_type', 'custom')
    
    if not shape_names:
        return jsonify({'error': 'No shapes selected'}), 400
    
    if state.first_frame is None:
        return jsonify({'error': 'No video loaded'}), 400
    
    if not state.video_name:
        return jsonify({'error': 'No video loaded'}), 400
    
    # Clear existing trackers
    state.trackers = {}
    
    # Get shapes for this video
    shapes_dict = state.shapes_manager.get_shapes_for_video(state.video_name)
    
    # Initialize trackers for each selected shape
    for idx, shape_name in enumerate(shape_names):
        if shape_name not in shapes_dict:
            continue
        
        points = shapes_dict[shape_name]
        color = state.colors[idx % len(state.colors)]
        
        # Create appropriate tracker
        if tracker_type == 'csrt':
            tracker_obj = AnnotationTrackerCSRT(state.first_frame, points)
        else:
            tracker_obj = AnnotationTracker(state.first_frame, points)
        
        state.trackers[shape_name] = {
            'tracker': tracker_obj,
            'color': color,
            'name': shape_name
        }
    
    # Reset video to beginning
    state.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    state.current_frame = 0
    
    print(f"Initialized {len(state.trackers)} trackers")
    
    return jsonify({
        'success': True,
        'tracker_count': len(state.trackers)
    })

@app.route('/api/play', methods=['POST'])
def play_video():
    if state.video_capture is None:
        return jsonify({'error': 'No video loaded'}), 400
    
    state.is_playing = True
    return jsonify({'success': True})

@app.route('/api/pause', methods=['POST'])
def pause_video():
    state.is_playing = False
    return jsonify({'success': True})

@app.route('/api/reset', methods=['POST'])
def reset_video():
    if state.video_capture:
        state.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        state.current_frame = 0
        state.is_playing = False
        
        # Reinitialize trackers with first frame
        if state.trackers and state.first_frame is not None:
            shapes_dict = state.shapes_manager.get_shapes_for_video(state.video_name)
            
            for shape_name, tracker_data in state.trackers.items():
                if shape_name in shapes_dict:
                    points = shapes_dict[shape_name]
                    
                    # Recreate tracker based on type
                    if isinstance(tracker_data['tracker'], AnnotationTrackerCSRT):
                        tracker_data['tracker'] = AnnotationTrackerCSRT(state.first_frame, points)
                    else:
                        tracker_data['tracker'] = AnnotationTracker(state.first_frame, points)
    
    return jsonify({'success': True})

import time

def generate_frames():
    print("📺 Video stream started")

    while True:
        # If no video, stop stream
        if state.video_capture is None:
            print("⛔ No video capture — closing stream")
            break

        # If paused, sleep (DO NOT spin)
        if not state.is_playing:
            time.sleep(0.03)
            continue

        # Read frame safely
        ret, frame = state.video_capture.read()

        if not ret:
            print("🔁 End of video — stopping playback")
            state.is_playing = False
            break   # IMPORTANT: close the MJPEG stream

        state.current_frame += 1

        # Apply tracking
        if state.trackers:
            for shape_name, tracker_data in state.trackers.items():
                tracker = tracker_data['tracker']
                color = tracker_data['color']
                name = tracker_data['name']

                pts, status, conf = tracker.update(frame)
                pts_int = pts.astype(int)

                cv2.polylines(frame, [pts_int], True, color, 2)

                for p in pts_int:
                    cv2.circle(frame, tuple(p), 4, color, -1)

                center = np.mean(pts_int, axis=0).astype(int)
                label = f"{name}: {status} ({conf:.2f})"
                cv2.putText(frame, label, tuple(center),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, f"Frame: {state.current_frame}/{state.total_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    print("🛑 Video stream closed")
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("=" * 70)
    print("HoloRay Motion-Tracked Annotation - Flask Backend")
    print("=" * 70)
    print("Server starting on http://localhost:5000")
    print("Default video path: Dataset/Lapchole/Lapchole1.mp4")
    print("=" * 70)
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)