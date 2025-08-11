import cv2
import os
import time
import numpy as np
import pandas as pd
import mediapipe as mp
from tensorflow.keras.models import load_model
import joblib

# === Load model and label encoder ===
model = load_model("/Users/ameermortaza/Desktop/YOLO-HandSign- Detection/Model (3).h5")
label_encoder = joblib.load("/Users/ameermortaza/Desktop/YOLO-HandSign- Detection/Model (3).pkl")

# === Setup MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# === Automatically detect the working camera ===
def find_working_camera(max_index=5):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print(f"✅ Using camera index: {i}")
                return i
    raise RuntimeError("❌ No working camera found.")

camera_index = find_working_camera()
cap = cv2.VideoCapture(camera_index)

# === Output CSV path ===
output_csv_path = "prediction_live.csv"
results = []

# === Timer setup ===
capture_interval = 3  # seconds
last_capture_time = time.time()

print("🟢 Starting webcam... Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame from webcam.")
        break

    frame = cv2.flip(frame, 1)  # Mirror effect for webcam
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    # Draw hand landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show live feed
    cv2.imshow("Webcam Feed (Press 'q' to quit)", frame)

    # === Every 3 seconds: capture frame and classify ===
    current_time = time.time()
    if current_time - last_capture_time >= capture_interval:
        last_capture_time = current_time

        if result.multi_hand_landmarks:
            keypoints = []
            for hand_landmarks in result.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])

            if len(result.multi_hand_landmarks) == 1:
                keypoints.extend([-1.0] * 63)

            if len(keypoints) == 126:
                X_input = np.array(keypoints).reshape(1, -1)
                prediction = model.predict(X_input)
                predicted_index = np.argmax(prediction)
                predicted_class = label_encoder.inverse_transform([predicted_index])[0]
                confidence = float(np.max(prediction))

                print(f"🖼️ Snapshot taken — Prediction: {predicted_class} ({confidence:.2f})")

                results.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "predicted_class": predicted_class,
                    "confidence": confidence
                })
            else:
                print("⚠️ Invalid keypoints length — skipping frame.")
        else:
            print("⚠️ No hand detected at capture time — skipping frame.")

    # === Quit on 'q' key ===
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Save to CSV ===
if results:
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"\n✅ Predictions saved to: {output_csv_path}")
    print(df.head())
else:
    print("\n⚠️ No predictions to save.")

cap.release()
cv2.destroyAllWindows()
print("🔴 Webcam stopped.")