import cv2
import mediapipe as mp
import csv
import os  #  Added to check if file exists

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)  # 🔧 max_num_hands=2

# Gesture Mapping (Key: Gesture)
GESTURE_MAP = {
    ord('1'): 'point',
    ord('2'): 'click',
    ord('3'): 'right_click',
    ord('4'): 'drag',
    ord('5'): 'scroll_up',
    ord('6'): 'scroll_down',
    ord('7'): 'screenshot'  #  Added screenshot gesture
}
current_gesture = 'point'  # Default gesture

#  Generate column headers dynamically for 2 hands (h0, h1)
def generate_headers():
    return ['label'] + [
        f'{axis}{i}_h{h}' for h in range(2) for i in range(21) for axis in ['x', 'y', 'z']
    ]

#  Write headers only if file doesn't exist
csv_file = 'gesture_data.csv'
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(generate_headers())

# Start video capture
cap = cv2.VideoCapture(0)
print("""
Gesture Controls:
1: point | 2: click | 3: right_click
4: drag | 5: scroll_up | 6: scroll_down | 7: screenshot
c: Save current frame | q: Quit
""")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror the frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Display current gesture
    cv2.putText(frame, f'Gesture: {current_gesture}', (20, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, 'Press 1-7 to change gesture', (20, 100),  # 🔧 Updated to 1-7
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Draw hand landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Gesture Data Collection', frame)

    # Keyboard controls
    key = cv2.waitKey(1)
    if key in GESTURE_MAP:
        current_gesture = GESTURE_MAP[key]
        print(f"Switched to gesture: {current_gesture}")
    elif key == ord('c'):
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            if current_gesture == 'screenshot' and num_hands < 2:
                print("[WARNING] Screenshot gesture requires TWO hands.")
                continue

            row = [current_gesture]
            for i in range(2):  # 🔧 Loop through both hands
                if i < num_hands:
                    hand_landmarks = results.multi_hand_landmarks[i]
                    row.extend([lm.x for lm in hand_landmarks.landmark])
                    row.extend([lm.y for lm in hand_landmarks.landmark])
                    row.extend([lm.z for lm in hand_landmarks.landmark])
                else:
                    row.extend([0.0] * 63)  # 🔧 Pad with zeros if hand missing

            with open(csv_file, 'a', newline='') as f:
                csv.writer(f).writerow(row)
            print(f"Saved sample for {current_gesture}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
