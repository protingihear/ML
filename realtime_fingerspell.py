#this will recognize the sign language you make, although you would need the model, and im not sure sharing it to a public repo is a great idea :)
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = load_model('hand_gesture_model.h5')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

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

# Label encoder for predictions (adjust this according to your dataset)
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
label_encoder = LabelEncoder()
label_encoder.fit(labels)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame color to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        hands_landmarks = []
        
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hands_landmarks.append(hand_landmarks)

        # If two hands are detected, combine the landmarks for both hands
        if len(hands_landmarks) == 2:
            hand_1 = preprocess_landmarks(hands_landmarks[0].landmark)
            hand_2 = preprocess_landmarks(hands_landmarks[1].landmark)
            combined_landmarks = hand_1 + hand_2  # Combine both hands' landmarks

            # Pad to ensure input size is consistent
            if len(combined_landmarks) < 126:
                combined_landmarks.extend([0] * (126 - len(combined_landmarks)))

            input_data = np.array([combined_landmarks])

            # Predict using the combined landmarks for two-hand gestures
            prediction = model.predict(input_data)
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

            # Display the predicted label for two-hand gestures
            cv2.putText(frame, f'Gesture: {predicted_label[0]} (Two Hands)', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # If only one hand is detected, predict based on the single hand
        elif len(hands_landmarks) == 1:
            hand_1 = preprocess_landmarks(hands_landmarks[0].landmark)

            # Pad the input for the second hand (if needed)
            input_data = np.array([hand_1 + [0] * 63])  # Pad with zeros for missing second hand

            # Predict using the single hand
            prediction = model.predict(input_data)
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

            # Display the predicted label for the single hand
            cv2.putText(frame, f'Gesture: {predicted_label[0]} (One Hand)', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Hand Gesture Detection', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()