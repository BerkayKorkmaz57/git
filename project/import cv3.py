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

# --- Load ML model and preprocessors ---
try:
    model = joblib.load('gesture_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    ml_enabled = True
    print("ML model loaded successfully!")
except:
    ml_enabled = False
    print("Using fallback gesture detection")

def get_landmark_features_both_hands(results):
    """
    Extract features for both hands if present.
    Returns 126 features (2 hands * 21 landmarks * 3 coords)
    Pads with zeros if one hand missing.
    """
    features = []
    hands_detected = results.multi_hand_landmarks or []
    
    # We expect max 2 hands — order is not guaranteed, but for now use detected order
    for i in range(2):
        if i < len(hands_detected):
            hand_landmarks = hands_detected[i]
            features.extend([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        else:
            # pad with zeros for missing hand
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
        # Draw landmarks for all detected hands
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Move cursor based on **index finger tip of first detected hand**
        first_hand = results.multi_hand_landmarks[0]
        index_tip = first_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x, y = int(index_tip.x * screen_width), int(index_tip.y * screen_height)
        pyautogui.moveTo(x, y, duration=0.05)

        if ml_enabled:
            features = scaler.transform([get_landmark_features_both_hands(results)])
            prediction_probs = model.predict_proba(features)[0]
            current_confidence = get_confidence(prediction_probs)

            if current_confidence >= confidence_threshold * 100:
                gesture_idx = np.argmax(prediction_probs)
                current_gesture = le.inverse_transform([gesture_idx])[0]
        else:
            # Fallback for single hand (simplified)
            thumb_tip = first_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_index_dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
            current_gesture = "drag" if thumb_index_dist < 0.05 else "click" if index_tip.y < thumb_tip.y else "point"
            current_confidence = 100.0

        # Display gesture and confidence
        cv2.putText(frame, f'Gesture: {current_gesture or "unknown"} ({current_confidence}%)', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        if current_confidence < confidence_threshold * 100:
            cv2.putText(frame, 'LOW CONFIDENCE - RELEASING ACTIONS', (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        # No hands detected — release drag if active
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
        # Low confidence or no gesture — release drag if dragging
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
