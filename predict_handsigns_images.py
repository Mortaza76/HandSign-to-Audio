# predict_handsigns.py

import cv2
import os
import numpy as np
import pandas as pd
import mediapipe as mp
from tensorflow.keras.models import load_model
import joblib

# Paths
model_path = "/Users/ameermortaza/Desktop/FINAL-PROJECT/Model (3).h5"
label_encoder_path = "/Users/ameermortaza/Desktop/FINAL-PROJECT/Model (3).pkl"
image_folder = "test_images"
output_csv_path = "prediction_images.csv"

# Load model and label encoder
model = load_model(model_path)
label_encoder = joblib.load(label_encoder_path)

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

results = []

# Loop through all images
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"❌ Could not read image: {filename}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

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

                results.append({
                    "filename": filename,
                    "predicted_class": predicted_class,
                    "confidence": confidence
                })
            else:
                print(f"Skipped {filename}: Invalid keypoints length = {len(keypoints)}")
        else:
            print(f"Skipped {filename}: No hands detected")

df = pd.DataFrame(results)
df.to_csv(output_csv_path, index=False)
print(f"✅ Predictions saved to {output_csv_path}")