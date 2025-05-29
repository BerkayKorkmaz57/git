import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

# Gesture Mapping
GESTURE_MAP = {
    ord('1'): 'point',
    ord('2'): 'click',
    ord('3'): 'right_click',
    ord('4'): 'drag',
    ord('5'): 'scroll_up',
    ord('6'): 'scroll_down',
    ord('7'): 'screenshot'
}
current_gesture = 'point'

# Generate headers for CSV
def generate_headers():
    return ['label'] + [
        f'{axis}{i}_h{h}' for h in range(2) for i in range(21) for axis in ['x', 'y', 'z']
    ]

# CSV setup
csv_file = 'gesture_data.csv'
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(generate_headers())

# Start video
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

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Overlay current gesture
    cv2.putText(frame, f'Gesture: {current_gesture}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, 'Press 1-7 to change gesture', (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Process hands
    handedness_info = []
    hand_data = {}

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_handedness in enumerate(results.multi_handedness):
            label = hand_handedness.classification[0].label  # 'Left' or 'Right'
            handedness_info.append(label)
            hand_data[label] = results.multi_hand_landmarks[idx]
            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[idx], mp_hands.HAND_CONNECTIONS)

    # Show hand type on screen
    if handedness_info:
        hand_label_text = " & ".join(handedness_info)
        cv2.putText(frame, f'Hand(s): {hand_label_text}', (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Gesture Data Collection', frame)

    # Handle key input
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
            for expected in ['Right', 'Left']:
                if expected in hand_data:
                    lm = hand_data[expected]
                    row.extend([pt.x for pt in lm.landmark])
                    row.extend([pt.y for pt in lm.landmark])
                    row.extend([pt.z for pt in lm.landmark])
                else:
                    row.extend([0.0] * 63)  # pad

            with open(csv_file, 'a', newline='') as f:
                csv.writer(f).writerow(row)
            print(f"Saved sample for {current_gesture} | Hands: {handedness_info}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
