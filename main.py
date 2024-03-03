import cv2 
import mediapipe as mp
import time

# Starting Vars
time_function_done = 0
counter = 0

# Init Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Method of capture
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('name-of-video')

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        if (time_function_done + 5) < time.time():
            time_function_done = time.time()
            counter += 1
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Recognition', frame)

    print(counter)

    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()        