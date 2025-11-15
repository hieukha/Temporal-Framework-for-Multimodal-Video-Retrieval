import os
import shutil
from concurrent.futures import ThreadPoolExecutor

base_dir = "/dataset/AIC_2025/SIU_Sayan/video"

def move_videos(folder_path):
    video_subdir = os.path.join(folder_path, "video")
    if os.path.exists(video_subdir):
        # Di chuyển tất cả file video lên thư mục cha
        for video_file in os.listdir(video_subdir):
            source_path = os.path.join(video_subdir, video_file)
            if os.path.isfile(source_path) and video_file.endswith((".mp4", ".avi", ".mov")):
                dest_path = os.path.join(folder_path, video_file)
                shutil.move(source_path, dest_path)
        # Xóa thư mục video nếu nó trống
        if not os.listdir(video_subdir):
            os.rmdir(video_subdir)

# Duyệt qua tất cả các thư mục Videos_L** và sử dụng ThreadPoolExecutor
folders = [os.path.join(base_dir, folder) for folder in os.listdir(base_dir) 
           if os.path.isdir(os.path.join(base_dir, folder)) and folder.startswith("Videos_L")]

with ThreadPoolExecutor(max_workers=32) as executor:
    executor.map(move_videos, folders)