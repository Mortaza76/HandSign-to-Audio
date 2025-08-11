import os
import asyncio
from gtts import gTTS
from googletrans import Translator
import edge_tts

# === Step 1: Read the sentence from the file ===
with open("generate_sentences_video.txt", "r") as file:
    sentence = file.read().strip()

# === Step 2: Translate to Chinese ===
translator = Translator()
translation = translator.translate(sentence, src="en", dest="zh-cn")
chinese_text = translation.text

# === Step 3: Create folder for audio output ===
output_folder = "output_audio"
os.makedirs(output_folder, exist_ok=True)

# === Step 4: Generate English audio using gTTS ===
english_audio_path = os.path.join(output_folder, "sentence_en.mp3")
gTTS(text=sentence, lang="en").save(english_audio_path)

# === Step 5: Generate Chinese audio using edge-tts ===
async def generate_chinese_audio(text, path):
    communicate = edge_tts.Communicate(text=text, voice="zh-CN-XiaoxiaoNeural")
    await communicate.save(path)

chinese_audio_path = os.path.join(output_folder, "sentence_zh.mp3")

# Run async function
if not asyncio.get_event_loop().is_running():
    asyncio.run(generate_chinese_audio(chinese_text, chinese_audio_path))
else:
    # For environments where an event loop is already running
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.get_event_loop().run_until_complete(generate_chinese_audio(chinese_text, chinese_audio_path))

# === Step 6: Print Paths ===
print(f"✅ English audio saved to: {english_audio_path}")
print(f"✅ Chinese audio saved to: {chinese_audio_path}")
