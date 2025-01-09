#this is just for visualization, later the landmarks and points marked in the video will be used as learning data as json sequences of each frame and for each label based on the folder the video is placed upon.
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

# Open video file
cap = cv2.VideoCapture('PUT THE VIDEO PATH HERE BOY')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

def draw_sub_points(image, point1, point2, num_points=5, color=(255, 255, 0)):
    """Visual helper only: Draws sub-points along arms."""
    if point1 and point2:
        for i in range(1, num_points):
            x = int(point1[0] + (point2[0] - point1[0]) * i / num_points)
            y = int(point1[1] + (point2[1] - point1[1]) * i / num_points)
            cv2.circle(image, (x, y), 3, color, -1)

def extract_landmark_coords(landmarks, indices, width, height):
    """Extracts (x, y) coordinates for a list of landmark indices."""
    coords = []
    for idx in indices:
        lm = landmarks[idx]
        coords.append((int(lm.x * width), int(lm.y * height)))
    return coords

# Variables to store last known hand landmarks
last_hand_landmarks = [None] * 2  # Ensure only two slots for hands
last_detection_time = [0] * 2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR frame to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with Mediapipe
    pose_results = pose.process(frame_rgb)
    hand_results = hands.process(frame_rgb)

    height, width, _ = frame.shape
    current_time = time.time()

    # Processing hands with safety check for index
    if hand_results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            if idx < len(last_hand_landmarks):  # Prevent index error
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=HAND_LANDMARK_STYLE,
                    connection_drawing_spec=HAND_CONNECTION_STYLE
                )
                last_hand_landmarks[idx] = hand_landmarks
                last_detection_time[idx] = current_time

    # Fallback to last known hand positions
    else:
        for idx in range(len(last_hand_landmarks)):
            if last_hand_landmarks[idx] and current_time - last_detection_time[idx] < 1:
                mp_drawing.draw_landmarks(
                    frame, last_hand_landmarks[idx], mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=HAND_LANDMARK_STYLE,
                    connection_drawing_spec=HAND_CONNECTION_STYLE
                )

    # Draw Pose landmarks and sub-points
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS, POSE_LANDMARK_STYLE)
        landmarks = pose_results.pose_landmarks.landmark

        left_arm_points = extract_landmark_coords(
            landmarks, [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST], width, height)
        right_arm_points = extract_landmark_coords(
            landmarks, [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST], width, height)

        draw_sub_points(frame, *left_arm_points[:2])
        draw_sub_points(frame, *left_arm_points[1:])
        draw_sub_points(frame, *right_arm_points[:2])
        draw_sub_points(frame, *right_arm_points[1:])

    # Write the frame to the output video
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

# Play the output video
def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Processed Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

play_video('output.mp4')
