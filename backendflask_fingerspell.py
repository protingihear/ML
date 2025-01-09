#simple script to get an input from a source and then it will be sent back the output of what hands did the model recognize it to be.
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model
model = load_model('hand_gesture_model.h5')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to normalize landmarks
def normalize_landmarks(landmarks):
    landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    min_vals = landmarks_np.min(axis=0)
    max_vals = landmarks_np.max(axis=0)
    normalized_landmarks = (landmarks_np - min_vals) / (max_vals - min_vals)
    return normalized_landmarks

# Function to preprocess hand landmarks
def preprocess_landmarks(landmarks):
    features = []
    normalized_landmarks = normalize_landmarks(landmarks)
    features.extend(normalized_landmarks.flatten())  # Flatten to 1D
    return features

# Label encoder for predictions, dataset A-Z aj
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
label_encoder = LabelEncoder()
label_encoder.fit(labels)

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the image file
    try:
        image_np = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Convert the image to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Process the landmarks
        if results.multi_hand_landmarks:
            hands_landmarks = []

            for hand_landmarks in results.multi_hand_landmarks:
                hands_landmarks.append(hand_landmarks)

            if len(hands_landmarks) == 2:  # Two hands detected
                hand_1 = preprocess_landmarks(hands_landmarks[0].landmark)
                hand_2 = preprocess_landmarks(hands_landmarks[1].landmark)
                combined_landmarks = hand_1 + hand_2

                # Pad to ensure input size is consistent
                if len(combined_landmarks) < 126:
                    combined_landmarks.extend([0] * (126 - len(combined_landmarks)))

                input_data = np.array([combined_landmarks])
                prediction = model.predict(input_data)
                predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

                return jsonify({'gesture': predicted_label[0], 'hands': 'Two Hands'})
            elif len(hands_landmarks) == 1:  # Single hand detected
                hand_1 = preprocess_landmarks(hands_landmarks[0].landmark)

                # Pad the input for the second hand
                input_data = np.array([hand_1 + [0] * 63])
                prediction = model.predict(input_data)
                predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

                return jsonify({'gesture': predicted_label[0], 'hands': 'One Hand'})
        else:
            return jsonify({'gesture': 'No hands detected!'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
