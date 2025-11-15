from tqdm import tqdm
import os
import numpy as np
import signal
import sys
from Class.DFN5B import DFN5B

# Biến để kiểm soát việc dừng
stop_processing = False

def signal_handler(sig, frame):
    global stop_processing
    print('\n\nNhận tín hiệu dừng. Đang hoàn thành xử lý video hiện tại và dừng an toàn...')
    stop_processing = True

# Đăng ký signal handler
signal.signal(signal.SIGINT, signal_handler)

model = DFN5B()

main_path = "/dataset/AIC_2025/SIU_Sayan/keyframes_filter"  # Đường dẫn mới
SAVE_DIR = "/dataset/AIC_2025/SIU_Sayan/autoshot/features_dfn5b_filter"
os.makedirs(SAVE_DIR, exist_ok=True)

# Đếm tổng số video cần xử lý và đã xử lý
total_videos = 0
processed_videos = 0
for level_folder in sorted(os.listdir(main_path)):
    level_folder_path = os.path.join(main_path, level_folder)
    if not os.path.isdir(level_folder_path) or level_folder == "SceneJson":
        continue
    keyframes_path = os.path.join(level_folder_path, "keyframes")
    if os.path.exists(keyframes_path):
        total_videos += len([d for d in os.listdir(keyframes_path) if os.path.isdir(os.path.join(keyframes_path, d))])

print(f"Tổng số video cần xử lý: {total_videos}")

# Duyệt qua các thư mục level (Keyframes_L21, Keyframes_L22, ...)
for level_folder in tqdm(sorted(os.listdir(main_path))):  # Keyframes_L21, Keyframes_L22, ...
    if stop_processing:
        print("Đã dừng xử lý theo yêu cầu.")
        break
        
    level_folder_path = os.path.join(main_path, level_folder)
    if not os.path.isdir(level_folder_path) or level_folder == "SceneJson":
        continue
    
    print(f"Processing level: {level_folder}")
    
    # Vào thư mục keyframes bên trong
    keyframes_path = os.path.join(level_folder_path, "keyframes")
    if not os.path.exists(keyframes_path):
        print(f"Keyframes folder not found in {level_folder}")
        continue
    
    # Duyệt qua các thư mục video trong keyframes (L21_V001, L21_V002, ...)
    for video_folder in tqdm(sorted(os.listdir(keyframes_path)), desc=f"Processing {level_folder}"):
        if stop_processing:
            print("Đã dừng xử lý theo yêu cầu.")
            break
            
        video_folder_path = os.path.join(keyframes_path, video_folder)
        if not os.path.isdir(video_folder_path):
            continue
        
        # Kiểm tra xem file đã được xử lý chưa
        save_name = f"{video_folder}.npy"
        save_path = os.path.join(SAVE_DIR, save_name)
        
        if os.path.exists(save_path):
            processed_videos += 1
            print(f"Đã bỏ qua {video_folder} (đã xử lý trước đó). Tiến độ: {processed_videos}/{total_videos}")
            continue
        
        result = []
        # Duyệt qua các file ảnh trong thư mục video
        for keyframe in tqdm(sorted(os.listdir(video_folder_path)), desc=f"Processing {video_folder}", leave=False):
            if stop_processing:
                print("Đã dừng xử lý theo yêu cầu.")
                break
                
            if not keyframe.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            keyframe_path = os.path.join(video_folder_path, keyframe)
            feet = model.get_image_features(keyframe_path)
            result.append(feet)
        
        if stop_processing:
            break
            
        # Lưu file theo tên thư mục video, ví dụ: L01_V001.npy
        np.save(save_path, result)
        processed_videos += 1
        print(f"Saved {video_folder}: {len(result)} features. Tiến độ: {processed_videos}/{total_videos}")

if stop_processing:
    print(f"\nĐã dừng xử lý. Đã hoàn thành {processed_videos}/{total_videos} video.")
else:
    print(f"\nHoàn thành tất cả! Đã xử lý {processed_videos}/{total_videos} video.")