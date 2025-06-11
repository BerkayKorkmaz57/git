import cv2
import mediapipe as mp
import pyautogui
import time
import joblib
import numpy as np
import datetime
from scipy.special import softmax

screenshot_cooldown = 2  # Cooldown for screenshot in seconds
last_screenshot_time = 0  # Last screenshot time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Screen dimensions
screen_width, screen_height = pyautogui.size()
dragging = False
last_click_time = 0
confidence_threshold = 0.7  # Minimum confidence to accept prediction (70%)

# Load ML model (if available)
try:
    model = joblib.load('gesture_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    ml_enabled = True
    print("ML model loaded successfully!")
except:
    ml_enabled = False
    print("Using fallback gesture detection")

def get_landmark_features(hand_landmarks):
    """Convert hand landmarks to feature vector"""
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

def get_confidence(prediction_probs):
    """Convert prediction probabilities to confidence percentage"""
    return round(max(prediction_probs) * 100, 2)

# Main loop
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_gesture = None
    current_confidence = 0
    
      # --- Two-hand gesture detection and screenshot ---
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        now = time.time()
        if now - last_screenshot_time > screenshot_cooldown:
            screenshot = pyautogui.screenshot()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            screenshot.save(filename)
            last_screenshot_time = now
            cv2.putText(frame, f'SCREENSHOT SAVED: {filename}', (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(frame, 'TWO HANDS DETECTED!', (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Virtual Mouse", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue  # Skip rest of loop to avoid mouse actions

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index finger position for cursor
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_tip.x * screen_width), int(index_tip.y * screen_height)
            pyautogui.moveTo(x, y, duration=0.05)

            # Gesture detection
            if ml_enabled:
                features = scaler.transform([get_landmark_features(hand_landmarks)])
                prediction_probs = model.predict_proba(features)[0]
                current_confidence = get_confidence(prediction_probs)
                
                if current_confidence >= confidence_threshold * 100:
                    gesture_idx = np.argmax(prediction_probs)
                    current_gesture = le.inverse_transform([gesture_idx])[0]
            else:
                # Simplified fallback (no scroll)
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                thumb_index_dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
                current_gesture = "drag" if thumb_index_dist < 0.05 else "click" if index_tip.y < thumb_tip.y else "point"
                current_confidence = 100.0  # Fallback is always 100% confident

            # Display gesture and confidence
            cv2.putText(frame, f'Gesture: {current_gesture or "unknown"} ({current_confidence}%)', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            if current_confidence < confidence_threshold * 100:
                cv2.putText(frame, 'LOW CONFIDENCE - RELEASING ACTIONS', (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Action handling (outside hand detection loop)
    if current_gesture and current_confidence >= confidence_threshold * 100:
        if current_gesture == "click" and time.time() - last_click_time > 0.3:
            pyautogui.click()
            last_click_time = time.time()
        elif current_gesture == "right_click" and time.time() - last_click_time > 0.3:
            pyautogui.rightClick()
            last_click_time = time.time()
        elif current_gesture == "drag":
            if not dragging:
                pyautogui.mouseDown()
                dragging = True
        elif current_gesture in ["scroll_up", "scroll_down"] and ml_enabled:
            pyautogui.scroll(100 if current_gesture == "scroll_up" else -100)
    else:
        # Release all actions if confidence is low
        if dragging:
            pyautogui.mouseUp()
            dragging = False

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        if dragging:  # Cleanup if quitting while dragging
            pyautogui.mouseUp()
        break

cap.release()
cv2.destroyAllWindows()