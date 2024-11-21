import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from flask import Flask, request, jsonify, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
expression_model = load_model('model_optimal.h5')
expression_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to detect and predict expressions
def detect_and_predict_expression(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    results = []

    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y + h, x:x + w]
        face_roi_resized = cv2.resize(face_roi, (48, 48))
        face_roi_normalized = face_roi_resized / 255.0
        face_roi_reshaped = np.expand_dims(face_roi_normalized, axis=(0, -1))

        predictions = expression_model.predict(face_roi_reshaped)
        expression_index = np.argmax(predictions)
        expression_label = expression_labels[expression_index]
        confidence = predictions[0][expression_index]

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f'{expression_label} ({confidence:.2f})', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        results.append({'label': expression_label, 'confidence': float(confidence), 'box': [int(x), int(y), int(w), int(h)]})

    return frame, results

# API for live streaming
@app.route('/live')
def live_feed():
    def generate_frames():
        cap = cv2.VideoCapture(0)
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame, _ = detect_and_predict_expression(frame)
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# API for uploading and processing an image
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Load and process the image
        frame = cv2.imread(filepath)
        frame, results = detect_and_predict_expression(frame)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{file.filename}')
        cv2.imwrite(output_path, frame)

        return jsonify({
            'processed_image': output_path,
            'results': results
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
