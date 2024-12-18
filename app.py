import os
import cv2
from flask import Flask, render_template, request, redirect, Response, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Set up paths for uploaded files and detection results
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
DETECTION_FOLDER = 'static/detections'

# Load YOLO model
model_path = 'best.pt'
model = YOLO(model_path)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTION_FOLDER, exist_ok=True)

# Function to detect objects in an image file
def detect_objects_in_image(image_path, model):
    img = cv2.imread(image_path)
    results = model(img)
    img_with_detections = results[0].plot()
    output_image_path = os.path.join(DETECTION_FOLDER, 'detected_image.jpg')
    cv2.imwrite(output_image_path, img_with_detections)
    return 'detections/detected_image.jpg'

# Function to detect objects in a video file
def detect_objects_in_video(video_path=None, use_webcam=False, model=None):
    if use_webcam:
        cap = cv2.VideoCapture(0)  # Webcam
    else:
        cap = cv2.VideoCapture(video_path)  # Video

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_file = os.path.join(DETECTION_FOLDER, 'detected_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        frame = results[0].plot()
        out.write(frame)
    cap.release()
    out.release()
    return 'detections/detected_video.mp4'

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html', image_path=None, video_path=None)

# Route to handle file uploads and detections
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    # Save the uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Handle image files
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        output_image = detect_objects_in_image(image_path=filepath, model=model)
        return render_template('index.html', image_path=output_image, video_path=None)

    # Handle video files
    elif file.filename.lower().endswith('.mp4'):
        output_video = detect_objects_in_video(video_path=filepath, use_webcam=False, model=model)
        return render_template('index.html', image_path=None, video_path=output_video)

    return "Unsupported file type", 400

# Route for webcam real-time detection
@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

# Stream webcam video with real-time detection
def gen(model):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        frame = results[0].plot()
        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(model), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)