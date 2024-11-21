from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load pre-trained face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load expression detection model
model = load_model('model_optimal.h5')  # Path to your model file
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']  # Update as per your model's labels

def detect_and_resize_face(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # If a face is found, process it
    if len(faces) > 0:
        # Assume we use the first detected face
        x, y, w, h = faces[0]
        face_region = gray[y:y+h, x:x+w]
        
        # Resize the face to 48x48 pixels
        face_resized = cv2.resize(face_region, (48, 48))
        return face_resized
    else:
        return None

def predict_expression(face_image):
    # Preprocess the image for the model
    face_image = face_image.astype('float32') / 255.0  # Normalize to 0-1
    face_image = img_to_array(face_image)  # Convert to tensor
    face_image = np.expand_dims(face_image, axis=0)  # Add batch dimension

    # Predict the expression
    predictions = model.predict(face_image)
    predicted_label = emotion_labels[np.argmax(predictions)]
    return predicted_label

@app.route('/face-detection', methods=['POST'])
def face_detection():
    # Get the image file from the request
    file = request.files.get('image')

    if file is None:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Open the file as an image (Pillow handles various formats)
        img = Image.open(file.stream)
        
        # Convert the image to a numpy array (OpenCV format)
        img_np = np.array(img)
        
        # Ensure the image has 3 channels (convert grayscale to BGR if needed)
        if len(img_np.shape) == 2:  # Grayscale image
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        
        # Detect and resize face
        face = detect_and_resize_face(img_np)

        if face is None:
            return jsonify({"error": "No face detected"}), 400

        # Predict the expression
        expression = predict_expression(face)

        # Convert the numpy array to an image for debugging or further use (optional)
        pil_img = Image.fromarray(face)

        # Save the image to a BytesIO object
        byte_io = BytesIO()
        pil_img.save(byte_io, 'PNG')
        byte_io.seek(0)

        # Encode the processed face as base64 (optional for debugging or returning to the client)
        img_base64 = base64.b64encode(byte_io.read()).decode('utf-8')

        # Return the expression prediction and the processed face image (optional)
        return jsonify({
            "expression": expression,
            "face_image": img_base64
        }), 200

    except UnidentifiedImageError:
        return jsonify({"error": "Invalid image file"}), 400

if __name__ == '__main__':
    app.run(debug=True)
