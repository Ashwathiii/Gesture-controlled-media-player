import cv2
import mediapipe as mp
import os
import pygame

# Function to detect hand gestures using MediaPipe
# Function to detect hand gestures using MediaPipe
# Function to detect hand gestures using MediaPipe
def detect_gesture(hand_landmarks):
    # Extracting relevant landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Check if all five fingers are extended
    if thumb_tip.y < index_tip.y < middle_tip.y < ring_tip.y < pinky_tip.y:
        return "Pause"
    # Check if only the thumb is extended
    elif thumb_tip.y < index_tip.y and middle_tip.y < ring_tip.y and pinky_tip.y < ring_tip.y:
        return "Previous Track"
    # Check if only the index finger is extended
    elif thumb_tip.y > index_tip.y and middle_tip.y > ring_tip.y and pinky_tip.y > ring_tip.y:
        return "Next Track"
    else:
        return "No Gesture"






# Function to play music from a directory using pygame
def play_music(directory, current_track):
    pygame.mixer.init()
    pygame.mixer.music.load(os.path.join(directory, current_track))
    pygame.mixer.music.play()

# Specify the path to your music directory
music_directory = r"C:\Users\aswat\Desktop\DL PROJECTS\mediapipe project\MUSIC"

# Get a list of music tracks in the directory
music_tracks = os.listdir(music_directory)
current_track_index = 0

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    # Check if hands are present
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect gestures
            gesture = detect_gesture(hand_landmarks.landmark)

            # Perform actions based on gestures
            if gesture == "Pause":
                pygame.mixer.music.pause()
            elif gesture == "Previous Track":
                current_track_index = (current_track_index - 1) % len(music_tracks)
                play_music(music_directory, music_tracks[current_track_index])
            elif gesture == "Next Track":
                current_track_index = (current_track_index + 1) % len(music_tracks)
                play_music(music_directory, music_tracks[current_track_index])

            # Display the detected gesture
            cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Gesture-Controlled Music Player", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()

