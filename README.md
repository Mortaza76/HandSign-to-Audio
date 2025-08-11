Sign Language Translation System
Overview

This project is a comprehensive sign language translation system that captures hand gestures via webcam or images, interprets them using machine learning, generates natural language sentences, and provides audio output in both English and Chinese. The system is designed to bridge communication gaps for sign language users.

Key Features

Real-time Hand Sign Detection: Uses MediaPipe and a trained ML model to recognize sign language gestures from webcam input
Image-based Sign Recognition: Processes static images to detect and classify hand signs
Natural Language Generation: Uses Gemini AI to convert detected signs into grammatically correct sentences
Multilingual Audio Output: Generates spoken output in both English and Chinese (Mandarin)
User-friendly GUI: PyQt5-based interface for easy interaction
Data Logging: Records predictions with timestamps and confidence levels
System Components

1. Core Modules

Prediction Modules:
webcam_testing.py - Real-time sign detection from webcam
predict_handsigns_images.py - Sign detection from static images
predict_sign_language.py - Alternative image-based sign detection
Sentence Generation:
generate_sentence.py - Creates sentences from image-based predictions
generated_sentences_live.py - Generates sentences from live predictions
generate_sentences_images.py - Specialized image-based sentence generation
Audio Generation:
generate_audio.py - Converts text to speech in English and Chinese
Integrated audio generation in GUI.py
User Interface:
GUI.py - Main graphical interface for the system
2. Data Files

Model (3).h5 - Trained Keras model for sign recognition
Model (3).pkl - Label encoder for sign classification
prediction_live.csv - Stores live prediction results
prediction_images.csv - Stores image-based prediction results
generated_sentences.txt - Output sentences from live predictions
generate_sentences_images.txt - Output sentences from image predictions
Installation

Prerequisites:
Python 3.8 or higher
pip package manager
Install dependencies:
bash
pip install -r requirements.txt
Where requirements.txt should contain:
text
tensorflow>=2.0
mediapipe
opencv-python
PyQt5
gTTS
googletrans==4.0.0-rc1
edge-tts
google-generativeai
pandas
numpy
joblib
python-dotenv
nest_asyncio
API Keys:
Set up your Gemini API key as an environment variable:
bash
export GEMINI_API_KEY="your_api_key_here"
Usage

1. Real-time Webcam Processing


python webcam_testing.py
Press 'q' to quit
Predictions are saved to prediction_live.csv
2. Image-based Processing


python predict_handsigns_images.py
Place test images in the test_images folder
Results saved to prediction_images.csv
3. Graphical User Interface


python GUI.py
Features:

Start/Stop camera
Generate sentences from detected signs
Generate audio output in English and Chinese
4. Sentence Generation

For live predictions:


python generated_sentences_live.py
For image predictions:

python generate_sentences_images.py
5. Audio Generation

python generate_audio.py

Model Details

Architecture: Custom Keras model trained on hand sign data
Input: 126-dimensional hand landmark coordinates (x,y,z for 42 landmarks)
Output: 20-class classification of common signs
Classes: Includes signs for "Me", "Sleep", "Late", "Please", "Water", "Drink", etc.
Limitations

Requires good lighting conditions for accurate sign detection
Works best with clear, unambiguous hand gestures
Currently supports a limited vocabulary of signs
Audio generation requires internet connection for some services
Future Enhancements

Expand vocabulary with more signs
Add support for additional languages
Implement sentence history and correction
Develop mobile application version
Add training interface for custom signs


Acknowledgments

MediaPipe for hand tracking
Google Gemini for natural language generation
gTTS and Edge TTS for audio synthesis
TensorFlow/Keras for machine learning framework
