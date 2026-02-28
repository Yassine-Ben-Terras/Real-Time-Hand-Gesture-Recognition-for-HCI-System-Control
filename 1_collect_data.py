import cv2
import mediapipe as mp
import csv
import copy
import itertools

# --- CONFIGURATION ---
FILE_NAME = 'keypoints.csv'
# 0 = Fist (Volume Mute), 1 = Thumbs Up (Volume Up), 2 = Open Palm (Volume Down)
# You can add more later!

# --- SETUP MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)


# --- HELPER FUNCTIONS ---
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    # Convert relative coordinates (0-1) to pixel coordinates
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # 1. Convert to Relative Coordinates (The "Math" Step)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:  # The Wrist is our anchor (0,0)
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # 2. Convert to 1D List (Flatten)
    # Turn [[x,y], [x,y]] into [x, y, x, y...]
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # 3. Normalization (Scale Invariant)
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def logging_csv(number, landmark_list):
    if 0 <= number <= 9:
        with open(FILE_NAME, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return


# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)

print("Collecting Data... Press '0', '1', or '2' to save frames. Press 'q' to quit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip image for "mirror" effect
    image = cv2.flip(image, 1)
    debug_image = copy.deepcopy(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 1. Calculate pixel coordinates
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)

            # 2. Convert to relative/normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(landmark_list)

            # 3. Draw on screen (Visual Feedback)
            mp_drawing.draw_landmarks(
                debug_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # 4. Check for key presses to SAVE data
            key = cv2.waitKey(10)
            if 48 <= key <= 57:  # ASCII for keys '0' through '9'
                csv_label = key - 48
                logging_csv(csv_label, pre_processed_landmark_list)
                print(f"Saved frame for Class {csv_label}")

    cv2.imshow('Data Collection', debug_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()