# generate_sentence.py

import pandas as pd
import google.generativeai as genai
import re

# Load Gemini model
genai.configure(api_key="AIzaSyC4h_QLZOZMUzQRzemTcwPSfjdBQO1I2Ac")
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

def generate_single_sentence(words_list):
    words = ", ".join(words_list)
    prompt = (
        f"You are an emotionally intelligent sentence generator.\n"
        f"Use the following 3–5 words as inspiration: {words}.\n"
        f"Generate one short, natural-sounding sentence (8–12 words max).\n"
        f"Generate ONE and ONLY ONE grammatically correct sentence (8–12 words max).\n"
        f"Do not include explanations, multiple options, or repeated outputs.\n"
        f"End with appropriate punctuation (., ?, !). Return only the sentence.\n"
        f"Understand the intent behind the words, not just their surface form.\n"
        f"Choose the correct sentence style: request, statement, or question.\n"
        f"Be grammatically correct and use appropriate punctuation (!, ?, or .)\n"
    )
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        sentences = re.findall(r'[^.!?]*[.!?]', text)
        return sentences[0].strip() if sentences else text
    except Exception as e:
        print(f"Error generating sentence: {e}")
        return "ERROR"

# Load predicted words
df = pd.read_csv("prediction_images.csv")
predicted_words = df["predicted_class"].tolist()

# Generate and save sentence
sentence = generate_single_sentence(predicted_words)
with open("generate_sentences_images.txt", "w") as f:
    f.write(sentence + "\n")

print(f"✅ Sentence generated and saved to generate_sentences_images.txt\n\n🗣️ {sentence}")