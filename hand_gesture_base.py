import cv2
import mediapipe as mp
import math
from collections import deque


# MediaPipe setup

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Gesture smoothing
gesture_history = deque(maxlen=15)

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, _ = frame.shape
            lm = [(int(p.x * w), int(p.y * h)) for p in hand_landmarks.landmark]

            wrist = lm[0]

            # Distances from wrist to fingertips
            index_d  = distance(lm[8], wrist)
            middle_d = distance(lm[12], wrist)
            ring_d   = distance(lm[16], wrist)
            pinky_d  = distance(lm[20], wrist)

            # Reference distance (hand size normalization)
            palm_d = distance(lm[0], lm[9])

            fingers = [
                index_d  > palm_d * 1.2,
                middle_d > palm_d * 1.2,
                ring_d   > palm_d * 1.2,
                pinky_d  > palm_d * 1.2
            ]

            # Gesture definitions (NO THUMB)
            if fingers == [True, False, False, False]:
                gesture = "ONE_FINGER"
            elif fingers == [True, True, False, False]:
                gesture = "TWO_FINGERS"
            elif fingers == [True, True, True, True]:
                gesture = "OPEN_HAND"
            elif fingers == [False, False, False, False]:
                gesture = "FIST"

            if gesture:
                gesture_history.append(gesture)

                # Majority vote
                stable_gesture = max(
                    set(gesture_history),
                    key=gesture_history.count
                )

                cv2.putText(
                    frame,
                    stable_gesture,
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("Improved Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
