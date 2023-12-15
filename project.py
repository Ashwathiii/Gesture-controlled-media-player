import cv2
import mediapipe as mp
import os
import pygame
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Set threshold values for left and right swipes
left_swipe_threshold = 100
right_swipe_threshold = 100

# Set debounce delay in seconds
debounce_delay = 1.0

# Set the directory path for your music files
music_directory = r'C:\Users\aswat\Desktop\DL PROJECTS\mediapipe project\MUSIC'

# Get a list of music files in the directory
music_files = [file for file in os.listdir(music_directory) if file.endswith(('.mp3', '.mpeg', '.wav'))]

# Index of the currently playing song
current_song_index = 0

# Load the first song
pygame.mixer.init()
pygame.mixer.music.load(os.path.join(music_directory, music_files[current_song_index]))


# Initialize the video capture object
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or specify the camera index

# Initialize last song change time
last_song_change_time = time.time()

while cap.isOpened():
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        # Get landmarks for the first hand (assuming one hand is in the frame)
        hand_landmarks = results.multi_hand_landmarks[0]

        # Extract X-coordinate of index finger
        index_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])

        # Check for left swipe
        if index_x < left_swipe_threshold:
            # Play the previous song if debounce time has passed
            if time.time() - last_song_change_time > debounce_delay:
                current_song_index = (current_song_index - 1) % len(music_files)
                pygame.mixer.music.load(os.path.join(music_directory, music_files[current_song_index]))
                pygame.mixer.music.play()
                last_song_change_time = time.time()

        # Check for right swipe
        elif index_x > frame.shape[1] - right_swipe_threshold:
            # Play the next song if debounce time has passed
            if time.time() - last_song_change_time > debounce_delay:
                current_song_index = (current_song_index + 1) % len(music_files)
                pygame.mixer.music.load(os.path.join(music_directory, music_files[current_song_index]))
                pygame.mixer.music.play()
                last_song_change_time = time.time()

    # Display the frame
    cv2.imshow('Hand Tracking', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()


#index finger should be pointing to camera and then swipe accordingly