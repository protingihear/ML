#no saving data or anything, just running mediapipe functions.
import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize Mediapipe solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configure Mediapipe models
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)

# Drawing specifications
POSE_LANDMARK_STYLE = mp_drawing_styles.get_default_pose_landmarks_style()
HAND_LANDMARK_STYLE = mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2)
HAND_CONNECTION_STYLE = mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2)

# Open webcam
cap = cv2.VideoCapture(0)

def draw_sub_points(image, point1, point2, num_points=5, color=(255, 255, 0)):
    if point1 and point2:
        for i in range(1, num_points):
            x = int(point1[0] + (point2[0] - point1[0]) * i / num_points)
            y = int(point1[1] + (point2[1] - point1[1]) * i / num_points)
            cv2.circle(image, (x, y), 3, color, -1)

def extract_landmark_coords(landmarks, indices, width, height):
    """Extract (x, y) coordinates for a list of landmark indices."""
    coords = []
    for idx in indices:
        lm = landmarks[idx]
        coords.append((int(lm.x * width), int(lm.y * height)))
    return coords

# Variables to store last known hand landmarks
last_hand_landmarks = [None, None]
last_detection_time = [0, 0]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR frame to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with Mediapipe
    pose_results = pose.process(frame_rgb)
    hand_results = hands.process(frame_rgb)

    # Draw Pose landmarks
    height, width, _ = frame.shape
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=POSE_LANDMARK_STYLE
        )

        # Extract and draw sub-points for arms
        landmarks = pose_results.pose_landmarks.landmark
        left_arm_points = extract_landmark_coords(
            landmarks,
            [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST],
            width, height
        )
        right_arm_points = extract_landmark_coords(
            landmarks,
            [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST],
            width, height
        )

        draw_sub_points(frame, *left_arm_points[:2], num_points=5, color=(0, 255, 255))
        draw_sub_points(frame, *left_arm_points[1:], num_points=5, color=(0, 255, 255))
        draw_sub_points(frame, *right_arm_points[:2], num_points=5, color=(0, 255, 255))
        draw_sub_points(frame, *right_arm_points[1:], num_points=5, color=(0, 255, 255))

    # Draw Hand landmarks or last known hands
    current_time = time.time()
    if hand_results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=HAND_LANDMARK_STYLE,
                connection_drawing_spec=HAND_CONNECTION_STYLE
            )
            if idx < len(last_hand_landmarks):
                last_hand_landmarks[idx] = hand_landmarks
                last_detection_time[idx] = current_time
    else:
        for idx in range(len(last_hand_landmarks)):
            if last_hand_landmarks[idx] and current_time - last_detection_time[idx] < 1:
                mp_drawing.draw_landmarks(
                    frame,
                    last_hand_landmarks[idx],
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=HAND_LANDMARK_STYLE,
                    connection_drawing_spec=HAND_CONNECTION_STYLE
                )
            else:
                last_hand_landmarks[idx] = None

    # Display the result
    cv2.imshow('Pose and Hand Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
