#to debug whether the trained data was done right or not.

import cv2
import json
import numpy as np

# Function to load skeleton data from saved JSON files
def load_landmarks_from_json(filepath):
    with open(filepath, 'r') as f:
        landmarks = json.load(f)
    return landmarks

# Function to convert landmarks into a format that can be plotted with OpenCV
def landmarks_to_np(landmarks):
    return np.array([[lm['x'], lm['y']] for lm in landmarks])

# Function to draw hand landmarks on an empty image
def draw_hand_skeleton(image, landmarks, color=(0, 255, 0), thickness=2):
    # Connections between landmarks to form a skeleton
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
    ]

    # Draw circles at each landmark position
    for point in landmarks:
        cv2.circle(image, tuple(point), 5, color, cv2.FILLED)

    # Draw lines between connected landmarks
    for start, end in connections:
        cv2.line(image, tuple(landmarks[start]), tuple(landmarks[end]), color, thickness)

# Visualize hand skeletons from a single JSON file
def visualize_hand_skeleton_from_json(json_file):
    # Load the landmarks from the JSON file
    landmarks_data = load_landmarks_from_json(json_file)
    
    # Create a blank image to draw on
    image = np.zeros((500, 500, 3), dtype=np.uint8)

    # Check the number of hands and visualize accordingly
    if len(landmarks_data) == 2:
        # Two hands
        hand_1_landmarks = landmarks_to_np(landmarks_data[0])
        hand_2_landmarks = landmarks_to_np(landmarks_data[1])
        
        # Scale the landmarks to fit the image
        hand_1_landmarks = (hand_1_landmarks * 400 + 50).astype(int)
        hand_2_landmarks = (hand_2_landmarks * 400 + 50).astype(int)

        # Draw the skeletons for both hands
        draw_hand_skeleton(image, hand_1_landmarks, color=(0, 255, 0))  # Hand 1 in green
        draw_hand_skeleton(image, hand_2_landmarks, color=(255, 0, 0))  # Hand 2 in red
    elif len(landmarks_data) == 1:
        # One hand
        hand_landmarks = landmarks_to_np(landmarks_data[0])
        
        # Scale the landmarks to fit the image
        hand_landmarks = (hand_landmarks * 400 + 50).astype(int)

        # Draw the skeleton for the single hand
        draw_hand_skeleton(image, hand_landmarks, color=(0, 255, 0))  # Hand in green
    else:
        print("Error: JSON file does not contain valid landmark data.")
        return
    
    # Display the image with the skeletons
    cv2.imshow('Hand Skeleton', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# CEK DISINI, KASIH '#' BUAT YG G MAU DICEK
#json_file = SATU TANGAN
json_file = "D:\AAL\Coding\Tes\hand_landmarks\X\X_1_scale.json"

# Visualize the skeleton
visualize_hand_skeleton_from_json(json_file)