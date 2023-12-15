import cv2
import mediapipe as mp
import openai
import time

# Set your OpenAI GPT-3 API key
openai.api_key = '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'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize OpenAI GPT-3 parameters
gpt3_prompt_prefix = "Draw a scene where "
gpt3_api_key = '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'  # Add your OpenAI GPT-3 API key here

def recognize_gesture(frame):
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    # Check if hands are present
    if results.multi_hand_landmarks:
        # Extract hand landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            # Example: Draw lines connecting hand landmarks
            for landmark in hand_landmarks.landmark:
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        # Example: Assume drawing is converted to a textual description
        description = "a person drawing in the air."

        return description

    return None

def generate_sentence_with_openai(description):
    # Construct a prompt for GPT-3
    prompt = gpt3_prompt_prefix + description

    # Request a response from OpenAI GPT-3
    response = None
    while response is None or 'choices' not in response:
        try:
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                max_tokens=100
            )
        except openai.error.RateLimitError as e:
            # Wait for 20 seconds before retrying
            time.sleep(20)

    # Extract the generated sentence from the response
    sentence = response['choices'][0]['text'].strip()

    return sentence

# Open the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect gestures using MediaPipe
    description = recognize_gesture(frame)

    if description:
        # Generate a sentence using OpenAI GPT-3
        sentence = generate_sentence_with_openai(description)

        # Display the sentence on the frame
        cv2.putText(frame, sentence, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Air Drawing to Text", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()
