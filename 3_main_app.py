import cv2
import mediapipe as mp
import pickle
import copy
import time
import itertools
import warnings
import numpy as np 
from collections import deque, Counter
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL, CoCreateInstance, GUID

# Silence specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message='SymbolDatabase.GetPrototype')

# --- AUDIO IMPORTS ---
try:
    from pycaw.pycaw import IAudioEndpointVolume, IMMDeviceEnumerator, EDataFlow, ERole
except ImportError:
    print("Error importing pycaw. Please run: pip install --upgrade pycaw comtypes")
    exit()

# --- CONFIGURATION ---
MODEL_PATH = 'model.pkl'
CLASS_NAMES = {0: "Mute", 1: "Vol UP", 2: "Vol DOWN", 3: "Neutral"}

# --- UI COLOR SCHEME (Modern Dark Theme) ---
UI_BG = (18, 18, 22)           # Deep charcoal background
UI_ACCENT = (147, 51, 234)     # Purple accent
UI_ACCENT_LIGHT = (196, 118, 255)  # Light purple
UI_SUCCESS = (34, 197, 94)     # Green
UI_WARNING = (234, 179, 8)     # Amber
UI_DANGER = (239, 68, 68)      # Red
UI_TEXT_PRIMARY = (248, 250, 252)   # Almost white
UI_TEXT_SECONDARY = (148, 163, 184) # Light gray
UI_PANEL = (30, 30, 36)        # Slightly lighter than BG
UI_BORDER = (51, 51, 60)       # Border color

# --- SETUP VOLUME CONTROL ---
try:
    CLSID_MMDeviceEnumerator = GUID("{BCDE0395-E52F-467C-8E3D-C4579291692E}")
    enumerator = CoCreateInstance(CLSID_MMDeviceEnumerator, IMMDeviceEnumerator, CLSCTX_ALL)
    device = enumerator.GetDefaultAudioEndpoint(0, 1)
    interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    
    current_vol = volume.GetMasterVolumeLevelScalar()
    print(f"Volume Control Connected. Current Volume: {current_vol:.2f}")

except Exception as e:
    print(f"CRITICAL ERROR: Could not connect to Audio. Details: {e}")
    exit()

# --- LOAD MODEL ---
print("Loading model...")
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"ERROR: Could not find {MODEL_PATH}. Did you run 2_train_model.py?")
    exit()

# --- SETUP MEDIAPIPE ---
try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
except AttributeError:
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import drawing_styles as mp_drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# --- HELPER FUNCTIONS ---
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def draw_rounded_rectangle(img, pt1, pt2, color, thickness, radius=20):
    """Draw a rounded rectangle"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Draw filled rectangle
    if thickness == -1:
        # Main rectangle
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        
        # Circles for corners
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        # Draw border only
        # Lines
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        
        # Corners
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

def create_ui_overlay(img, gesture_name, confidence, volume_level, is_muted):
    """Create elegant UI overlay"""
    h, w = img.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:] = UI_BG
    
    # --- TOP STATUS BAR ---
    # Background panel
    draw_rounded_rectangle(overlay, (20, 20), (w - 20, 100), UI_PANEL, -1, 15)
    
    # Gesture status with indicator dot
    gesture_color = UI_ACCENT_LIGHT
    if gesture_name == "Mute":
        gesture_color = UI_DANGER
    elif gesture_name == "Vol UP":
        gesture_color = UI_SUCCESS
    elif gesture_name == "Vol DOWN":
        gesture_color = UI_WARNING
    
    # Status indicator dot
    cv2.circle(overlay, (50, 52), 8, gesture_color, -1)
    cv2.circle(overlay, (50, 52), 10, gesture_color, 2)
    
    # Gesture text
    cv2.putText(overlay, gesture_name.upper(), (75, 65),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, UI_TEXT_PRIMARY, 2, cv2.LINE_AA)
    
    # Confidence indicator with label
    conf_text = f"{int(confidence * 100)}"
    cv2.putText(overlay, "CONF", (w - 140, 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, UI_TEXT_SECONDARY, 1, cv2.LINE_AA)
    cv2.putText(overlay, f"{conf_text}%", (w - 135, 75),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, UI_TEXT_PRIMARY, 2, cv2.LINE_AA)
    
    # --- VOLUME CONTROL PANEL ---
    panel_x = w - 180
    panel_y = 140
    panel_w = 160
    panel_h = 300
    
    # Panel background
    draw_rounded_rectangle(overlay, (panel_x, panel_y), 
                          (panel_x + panel_w, panel_y + panel_h), 
                          UI_PANEL, -1, 20)
    
    # Volume bar container
    bar_x = panel_x + 50
    bar_y = panel_y + 40
    bar_w = 60
    bar_h = 180
    
    # Border
    draw_rounded_rectangle(overlay, (bar_x, bar_y), 
                          (bar_x + bar_w, bar_y + bar_h),
                          UI_BORDER, 2, 15)
    
    # Fill level
    if not is_muted:
        fill_height = int(bar_h * volume_level)
        fill_y = bar_y + bar_h - fill_height
        
        if fill_height > 4:
            # Gradient effect - create multiple shades
            for i in range(fill_height):
                ratio = i / fill_height if fill_height > 0 else 0
                # Blend from accent to accent_light
                color = tuple([
                    int(UI_ACCENT[j] + (UI_ACCENT_LIGHT[j] - UI_ACCENT[j]) * ratio)
                    for j in range(3)
                ])
                cv2.line(overlay, 
                        (bar_x + 2, fill_y + i), 
                        (bar_x + bar_w - 2, fill_y + i), 
                        color, 1)
    
    # Volume percentage
    if is_muted:
        vol_text = "MUTED"
        vol_color = UI_DANGER
    else:
        vol_text = f"{int(volume_level * 100)}"
        vol_color = UI_TEXT_PRIMARY
    
    # Center the percentage number
    text_size = cv2.getTextSize(vol_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0]
    text_x = panel_x + (panel_w - text_size[0]) // 2
    
    cv2.putText(overlay, vol_text, (text_x, panel_y + panel_h - 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, vol_color, 2, cv2.LINE_AA)
    
    # Add "%" symbol separately if not muted
    if not is_muted:
        cv2.putText(overlay, "%", (text_x + text_size[0] + 5, panel_y + panel_h - 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, UI_TEXT_SECONDARY, 2, cv2.LINE_AA)
    
    # Volume icon/label
    icon_label_y = panel_y + 20
    cv2.putText(overlay, "VOLUME", (panel_x + 35, icon_label_y),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, UI_TEXT_SECONDARY, 1, cv2.LINE_AA)
    
    # --- INSTRUCTIONS PANEL ---
    instr_y = h - 180
    draw_rounded_rectangle(overlay, (20, instr_y), (420, h - 20), UI_PANEL, -1, 15)
    
    # Title
    cv2.putText(overlay, "GESTURES", (40, instr_y + 30),
               cv2.FONT_HERSHEY_DUPLEX, 0.7, UI_TEXT_SECONDARY, 1, cv2.LINE_AA)
    
    instructions = [
        ("UP", "Volume Up", UI_SUCCESS),
        ("DN", "Volume Down", UI_WARNING),
        ("X", "Mute", UI_DANGER),
    ]
    
    y_offset = instr_y + 65
    for icon, text, color in instructions:
        # Draw icon box
        icon_box_x = 40
        draw_rounded_rectangle(overlay, (icon_box_x, y_offset - 20), 
                              (icon_box_x + 35, y_offset + 5), color, 2, 8)
        
        # Icon text
        cv2.putText(overlay, icon, (icon_box_x + 5, y_offset - 2),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        # Description
        cv2.putText(overlay, text, (icon_box_x + 50, y_offset - 2),
                   cv2.FONT_HERSHEY_DUPLEX, 0.65, UI_TEXT_PRIMARY, 1, cv2.LINE_AA)
        y_offset += 40
    
    return overlay

def draw_hand_skeleton(img, landmarks, gesture_id):
    """Draw elegant hand skeleton"""
    # Choose color based on gesture
    if gesture_id == 0:  # Mute
        connection_color = UI_DANGER
        landmark_color = UI_DANGER
    elif gesture_id == 1:  # Vol UP
        connection_color = UI_SUCCESS
        landmark_color = UI_SUCCESS
    elif gesture_id == 2:  # Vol DOWN
        connection_color = UI_WARNING
        landmark_color = UI_WARNING
    else:  # Neutral
        connection_color = UI_ACCENT_LIGHT
        landmark_color = UI_ACCENT_LIGHT
    
    # Draw connections with glow effect
    for connection in mp_hands.HAND_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        
        start_point = landmarks[start_idx]
        end_point = landmarks[end_idx]
        
        # Outer glow
        cv2.line(img, tuple(start_point), tuple(end_point), connection_color, 4)
        # Inner bright line
        cv2.line(img, tuple(start_point), tuple(end_point), (255, 255, 255), 1)
    
    # Draw landmarks
    for point in landmarks:
        # Outer glow
        cv2.circle(img, tuple(point), 6, connection_color, -1)
        # Inner bright dot
        cv2.circle(img, tuple(point), 3, (255, 255, 255), -1)

# --- STABILITY & SMOOTHING VARIABLES ---
last_action_time = 0   
action_cooldown = 0.15

history_length = 7
gesture_history = deque(maxlen=history_length)

# Animation variables
volume_animation = 0.5
animation_speed = 0.15

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("╔══════════════════════════════════════════╗")
print("║   GESTURE CONTROL SYSTEM INITIALIZED     ║")
print("║   Press 'Q' to quit                      ║")
print("╚══════════════════════════════════════════╝")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    h, w = image.shape[:2]
    
    # Process image
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    # Create UI overlay
    current_action = "Neutral"
    probability = 0.0
    most_common_gesture = 3
    
    # Get current volume for animation
    try:
        current_vol = volume.GetMasterVolumeLevelScalar()
        is_muted = volume.GetMute()
    except:
        current_vol = 0.5
        is_muted = False
    
    # Smooth volume animation
    target_vol = 0 if is_muted else current_vol
    volume_animation += (target_vol - volume_animation) * animation_speed
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Process Data
            landmark_list = calc_landmark_list(image, hand_landmarks)
            processed_data = pre_process_landmark(landmark_list)

            # Predict Raw Gesture
            prediction = model.predict([processed_data])
            raw_class_id = prediction[0]
            probability = model.predict_proba([processed_data])[0][raw_class_id]

            # Update History Buffer
            if probability > 0.8:
                gesture_history.append(raw_class_id)
            else:
                gesture_history.append(3)

            # Filter Glitches (Majority Vote)
            if len(gesture_history) > 0:
                most_common_gesture = Counter(gesture_history).most_common(1)[0][0]
            else:
                most_common_gesture = raw_class_id

            current_action = CLASS_NAMES.get(most_common_gesture, "Unknown")

            # VOLUME ACTION LOGIC ---
            if most_common_gesture == 0:
                volume.SetMute(1, None)

            elif most_common_gesture == 3:
                pass 

            elif time.time() - last_action_time > action_cooldown:
                try:
                    current_vol = volume.GetMasterVolumeLevelScalar()
                except:
                    current_vol = 0.5

                if most_common_gesture == 1:
                    volume.SetMute(0, None)
                    new_vol = min(current_vol + 0.02, 1.0)
                    volume.SetMasterVolumeLevelScalar(new_vol, None)
                    last_action_time = time.time()

                elif most_common_gesture == 2:
                    new_vol = max(current_vol - 0.02, 0.0)
                    volume.SetMasterVolumeLevelScalar(new_vol, None)
                    last_action_time = time.time()
            
            # Draw elegant hand skeleton on camera feed
            draw_hand_skeleton(image, landmark_list, most_common_gesture)
    
    # Create UI overlay
    ui_overlay = create_ui_overlay(image, current_action, probability, 
                                   volume_animation, is_muted)
    
    # Blend camera feed with UI
    alpha = 0.35  # Transparency of camera feed
    output = cv2.addWeighted(ui_overlay, 1 - alpha, image, alpha, 0)
    
    cv2.imshow('Gesture Control', output)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
