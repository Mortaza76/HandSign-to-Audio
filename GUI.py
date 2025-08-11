import sys
import os
import cv2
import csv
import time
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

import mediapipe as mp
from tensorflow.keras.models import load_model
import joblib

from gtts import gTTS
from googletrans import Translator
import edge_tts

# ========== Load Model and Label Encoder ==========
model = load_model("/Users/ameermortaza/Desktop/YOLO-HandSign- Detection/Model (3).h5")
label_encoder = joblib.load("/Users/ameermortaza/Desktop/YOLO-HandSign- Detection/Model (3).pkl")

# ========== MediaPipe Setup ==========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2
)
mp_drawing = mp.solutions.drawing_utils

# ========== Globals ==========
cap = None
captured_predictions = set()
data_dir = os.getcwd()
last_capture_time = 0
capture_interval = 3  # seconds

# ========== Feature Extraction ==========
def extract_keypoints(hand_landmarks):
    keypoints = []
    for lm in hand_landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.z])
    return keypoints

# ========== Sentence & Audio Generation ==========
def generate_sentence(predicted_word):
    sentence = f"This is the sign for {predicted_word}"
    with open("generated_sentences.txt", "w") as f:
        f.write(sentence)
    return sentence

def generate_audio(sentence):
    translator = Translator()
    translation = translator.translate(sentence, src="en", dest="zh-cn")
    chinese_text = translation.text

    en_tts = gTTS(text=sentence, lang="en")
    zh_tts = gTTS(text=chinese_text, lang="zh-CN")

    en_audio = "en_output.mp3"
    zh_audio = "zh_output.mp3"

    en_tts.save(en_audio)
    zh_tts.save(zh_audio)

    return en_audio, zh_audio

# ========== GUI Class ==========
class SignLanguageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Translator")

        # Layout
        self.video_label = QLabel()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.sentence_button = QPushButton("Generate Sentence")
        self.audio_button = QPushButton("Generate Audio")

        h_layout = QHBoxLayout()
        h_layout.addWidget(self.start_button)
        h_layout.addWidget(self.stop_button)
        h_layout.addWidget(self.sentence_button)
        h_layout.addWidget(self.audio_button)

        v_layout = QVBoxLayout()
        v_layout.addWidget(self.video_label)
        v_layout.addLayout(h_layout)
        self.setLayout(v_layout)

        # Events
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.sentence_button.clicked.connect(self.generate_sentence)
        self.audio_button.clicked.connect(self.generate_audio)

        # Timer for frame update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self):
        global cap
        cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def stop_camera(self):
        global cap
        self.timer.stop()
        if cap is not None:
            cap.release()
        self.video_label.clear()
        self.save_predictions()

    def update_frame(self):
        global last_capture_time
        success, frame = cap.read()
        if not success:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Every 3 seconds, capture a frame for prediction
                if time.time() - last_capture_time >= capture_interval:
                    keypoints = extract_keypoints(hand_landmarks)
                    if len(keypoints) == model.input_shape[1]:
                        prediction = model.predict(np.array([keypoints]))[0]
                        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

                        print(f"Predicted: {predicted_label}")
                        captured_predictions.add(predicted_label)
                        last_capture_time = time.time()

        image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format.Format_BGR888)
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def save_predictions(self):
        if captured_predictions:
            filename = os.path.join(data_dir, "prediction_live.csv")
            with open(filename, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Signs"])
                joined_signs = " ".join(captured_predictions)
                writer.writerow([joined_signs])
            print(f"Saved to {filename}")
        else:
            print("No valid predictions to save.")

    def generate_sentence(self):
        if captured_predictions:
            latest_sign = list(captured_predictions)[-1]
            sentence = generate_sentence(latest_sign)
            print("Sentence:", sentence)
        else:
            print("No signs to generate sentence from.")

    def generate_audio(self):
        if os.path.exists("generated_sentences.txt"):
            with open("generated_sentences.txt", "r") as file:
                sentence = file.read().strip()
                en_audio, zh_audio = generate_audio(sentence)
                print(f"English audio saved as: {en_audio}")
                print(f"Chinese audio saved as: {zh_audio}")
        else:
            print("No sentence found to generate audio.")

# ========== App Execution ==========
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec())