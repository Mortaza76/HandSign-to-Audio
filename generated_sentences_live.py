import pandas as pd
import google.generativeai as genai
import re
import os

# === Load API key ===
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# === Load Gemini model ===
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

# === Sentence generation function ===
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
        f"Examples:\n"
        f"  Words: Love Family → Sentence: I really love my family.\n"
        f"  Words: Eat Food Now → Sentence: Can we eat some food now?\n"
        f"  Words: Please Water → Sentence: Please give me some water.\n"
        f"  Words: You Where → Sentence: Where are you?\n"
        f"  Words: Smile Beautiful → Sentence: Your smile is beautiful!\n"
        f"Now generate the sentence:"
    )
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()

        sentences = re.findall(r'[^.!?]*[.!?]', text)
        return sentences[0].strip() if sentences else text

    except Exception as e:
        print(f"❌ Error generating sentence: {e}")
        return "ERROR"

# === Load CSV predictions ===
csv_path = "prediction_live.csv"

if not os.path.exists(csv_path):
    print("❌ Error: prediction_live.csv not found.")
    exit()

df = pd.read_csv(csv_path)

# === Extract unique valid words ===
if "predicted_class" not in df.columns:
    print("❌ Error: 'predicted_class' column not found in CSV.")
    exit()

predicted_words = df["predicted_class"].dropna().unique().tolist()

# === Check word count ===
if len(predicted_words) < 3:
    print("⚠️ Not enough unique predictions to form a sentence.")
    exit()

# === Generate sentence ===
sentence = generate_single_sentence(predicted_words)

# === Save to TXT ===
output_path = "generated_sentences.txt"
with open(output_path, "w") as f:
    f.write(sentence + "\n")

print(f"✅ Sentence generated and saved to {output_path}\n\n🗣️ {sentence}")