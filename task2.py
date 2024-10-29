import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
def is_letter_A(landmarks):
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    index_tip = landmarks[8]
    index_dip = landmarks[7]
    index_pip = landmarks[6]
    middle_tip = landmarks[12]
    middle_dip = landmarks[11]
    middle_pip = landmarks[10]
    ring_tip = landmarks[16]
    ring_dip = landmarks[15]
    ring_pip = landmarks[14]
    pinky_tip = landmarks[20]
    pinky_dip = landmarks[19]
    pinky_pip = landmarks[18]
    index_folded = index_tip.y > index_pip.y
    middle_folded = middle_tip.y > middle_pip.y
    ring_folded = ring_tip.y > ring_pip.y
    pinky_folded = pinky_tip.y > pinky_pip.y
    thumb_extended = thumb_tip.x < thumb_ip.x
    return index_folded and middle_folded and ring_folded and pinky_folded and thumb_extended
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmark_list = hand_landmarks.landmark
                if is_letter_A(landmark_list):
                    cv2.putText(frame, "Letter 'A' detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.imshow("Indian Sign Language Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

