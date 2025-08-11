import cv2
import os
import numpy as np
import pandas as pd
import mediapipe as mp
from tensorflow.keras.models import load_model
import joblib
from PIL import Image

# === Update These Paths ===
model_path = "/Users/ameermortaza/Desktop/FINAL-PROJECT/sign_language_model (2).h5"  # place this file in your project folder
label_encoder_path = "/Users/ameermortaza/Desktop/FINAL-PROJECT/label_encoder (1).pkl"  # place this file in your project folder
image_folder = "/Users/ameermortaza/Desktop/FINAL-PROJECT/test_images"  # a folder inside your project with test images
output_csv_path = "/Users/ameermortaza/Desktop/FINAL-PROJECT/prediction.csv"

# Load model and label encoder
model = load_model(model_path)
label_encoder = joblib.load(label_encoder_path)

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# Store predictions
results = []

# Loop through all images
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_path = os.path.join(image_folder, filename)

        # Read image and convert to RGB
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not read image: {filename}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        # Check if hands were detected
        if result.multi_hand_landmarks:
            keypoints = []

            for hand_landmarks in result.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])

            # If only one hand, pad to 126
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
                print(f"⚠️ Skipped {filename}: Invalid keypoints length = {len(keypoints)}")
        else:
            print(f"⚠️ Skipped {filename}: No hands detected")

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv(output_csv_path, index=False)

print(f"\n✅ Predictions saved to: {output_csv_path}")
print(df.head())
