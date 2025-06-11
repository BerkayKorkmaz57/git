import cv2 # Used for image processing and camera access.
import mediapipe as mp # Used to detect hand movements.t
import csv #Used to write data to files.

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Gesture Mapping (Key: Gesture)
GESTURE_MAP = {
    ord('1'): 'point',
    ord('2'): 'click',
    ord('3'): 'right_click',
    ord('4'): 'drag',
    ord('5'): 'scroll_up',
    ord('6'): 'scroll_down'
}
current_gesture = 'point'  # Default gesture

# Initialize CSV
with open('gesture_data.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    if f.tell() == 0:  # Write headers only if file is empty
        headers = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']] 
        writer.writerow(headers)

# Start video capture
cap = cv2.VideoCapture(0)
print("""
Gesture Controls:
1: point | 2: click | 3: right_click
4: drag | 5: scroll_up | 6: scroll_down
c: Save current frame | q: Quit
""")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror the frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #BGR -> RGB
    results = hands.process(rgb_frame) # detects hand movements.

    # Display current gesture
    cv2.putText(frame, f'Gesture: {current_gesture}', (20, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, 'Press 1-6 to change gesture', (20, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Draw hand landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks 
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Gesture Data Collection', frame)

    # Keyboard controls
    key = cv2.waitKey(1)
    if key in GESTURE_MAP:  # Change gesture
        current_gesture = GESTURE_MAP[key]
        print(f"Switched to gesture: {current_gesture}")
    elif key == ord('c'):  # Save data
        if results.multi_hand_landmarks:
            row = [current_gesture]
            for landmark in results.multi_hand_landmarks[0].landmark:
                row.extend([landmark.x, landmark.y, landmark.z])
            with open('gesture_data.csv', 'a', newline='') as f:
                csv.writer(f).writerow(row)
            print(f"Saved sample for {current_gesture}")
    elif key == ord('q'):  # Quit
        break

cap.release()
cv2.destroyAllWindows()