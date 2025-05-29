import cv2
import mediapipe as mp
import pyautogui
import time
import joblib
import numpy as np
import datetime

# --- Initialize MediaPipe Hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- Screen size for cursor mapping ---
screen_width, screen_height = pyautogui.size()

dragging = False
last_click_time = 0
confidence_threshold = 0.7  # Only accept predictions with 70%+ confidence

# --- Load both models and preprocessors ---
try:
    model_one = joblib.load('gesture_model_one_hand.pkl')
    scaler_one = joblib.load('scaler_one_hand.pkl')
    le_one = joblib.load('label_encoder_one_hand.pkl')

    model_two = joblib.load('gesture_model_two_hands.pkl')
    scaler_two = joblib.load('scaler_two_hands.pkl')
    le_two = joblib.load('label_encoder_two_hands.pkl')

    ml_enabled = True
    print("Both models loaded successfully!")
except:
    ml_enabled = False
    print("Using fallback gesture detection")

def get_landmark_features(hand_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

def get_landmark_features_both_hands(results):
    features = []
    hands_detected = results.multi_hand_landmarks or []
    for i in range(2):
        if i < len(hands_detected):
            features.extend([[lm.x, lm.y, lm.z] for lm in hands_detected[i].landmark])
        else:
            features.extend([[0, 0, 0]] * 21)
    return np.array(features).flatten()

def get_confidence(prediction_probs):
    return round(max(prediction_probs) * 100, 2)

# --- Main loop ---
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

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        first_hand = results.multi_hand_landmarks[0]
        index_tip = first_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x, y = int(index_tip.x * screen_width), int(index_tip.y * screen_height)
        pyautogui.moveTo(x, y, duration=0.05)

        if ml_enabled:
            if len(results.multi_hand_landmarks) == 1:
                features = scaler_one.transform([get_landmark_features(first_hand)])
                prediction_probs = model_one.predict_proba(features)[0]
                current_confidence = get_confidence(prediction_probs)
                if current_confidence >= confidence_threshold * 100:
                    gesture_idx = np.argmax(prediction_probs)
                    current_gesture = le_one.inverse_transform([gesture_idx])[0]
            else:
                features = scaler_two.transform([get_landmark_features_both_hands(results)])
                prediction_probs = model_two.predict_proba(features)[0]
                current_confidence = get_confidence(prediction_probs)
                if current_confidence >= confidence_threshold * 100:
                    gesture_idx = np.argmax(prediction_probs)
                    current_gesture = le_two.inverse_transform([gesture_idx])[0]
        else:
            thumb_tip = first_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_index_dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
            current_gesture = "drag" if thumb_index_dist < 0.05 else "click" if index_tip.y < thumb_tip.y else "point"
            current_confidence = 100.0

        cv2.putText(frame, f'Gesture: {current_gesture or "unknown"} ({current_confidence}%)', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        if current_confidence < confidence_threshold * 100:
            cv2.putText(frame, 'LOW CONFIDENCE - RELEASING ACTIONS', (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        if dragging:
            pyautogui.mouseUp()
            dragging = False

    # --- Action handling ---
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
        elif current_gesture == "screenshot":
            filename = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            pyautogui.screenshot(filename)
            print(f"Screenshot saved as {filename}")
    else:
        if dragging:
            pyautogui.mouseUp()
            dragging = False

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        if dragging:
            pyautogui.mouseUp()
        break

cap.release()
cv2.destroyAllWindows()
