from transformers import pipeline
import os
import json

AUDIO_ROOT = "/dataset/AIC_2025/SIU_Sayan/audio"
OUTPUT_JSON = "/dataset/AIC_2025/SIU_Sayan/audio/transcript_all.json"

asr = pipeline(model="openai/whisper-large-v2", device=0)  # device=0 nếu dùng GPU, -1 nếu CPU

results = {}

for root, dirs, files in os.walk(AUDIO_ROOT):
    for file in files:
        if file.endswith(".wav"):
            audio_path = os.path.join(root, file)
            print(f"Transcribing {audio_path} ...")
            prediction = asr(audio_path, chunk_length_s=30, stride_length_s=5, generate_kwargs={"task": "transcribe"})
            # Lưu theo đường dẫn tương đối để dễ mapping
            rel_path = os.path.relpath(audio_path, AUDIO_ROOT)
            results[rel_path] = prediction["text"]

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Done! Kết quả đã lưu vào", OUTPUT_JSON)