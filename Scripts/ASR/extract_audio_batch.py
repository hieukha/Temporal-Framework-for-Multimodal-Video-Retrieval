import os
import subprocess

# Đường dẫn thư mục chứa các video
VIDEO_ROOT = "/dataset/AIC_2025/SIU_Sayan/videos"
# Đường dẫn thư mục lưu audio output
AUDIO_ROOT = "/dataset/AIC_2025/SIU_Sayan/audio"

os.makedirs(AUDIO_ROOT, exist_ok=True)

for root, dirs, files in os.walk(VIDEO_ROOT):
    for file in files:
        if file.endswith(".mp4"):
            video_path = os.path.join(root, file)
            # Tạo đường dẫn thư mục con tương ứng trong AUDIO_ROOT
            rel_dir = os.path.relpath(root, VIDEO_ROOT)
            audio_dir = os.path.join(AUDIO_ROOT, rel_dir)
            os.makedirs(audio_dir, exist_ok=True)
            # Đặt tên file audio tương ứng
            audio_path = os.path.join(audio_dir, file.replace(".mp4", ".wav"))
            # Lệnh ffmpeg tách audio
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
            ]
            print("Extracting:", video_path, "->", audio_path)
            subprocess.run(cmd, check=True)