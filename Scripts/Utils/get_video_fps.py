import cv2
import os
import json
from tqdm import tqdm

def get_video_fps(video_path):
    """
    Lấy FPS của video
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return float(fps)
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return None

def extract_fps_from_directory(video_dir, output_json_path):
    """
    Duyệt qua tất cả video trong thư mục và lấy FPS
    """
    fps_dict = {}
    
    if not os.path.exists(video_dir):
        print(f"Directory not found: {video_dir}")
        return
    
    # Lấy tất cả file .mp4 trong thư mục
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    video_files.sort()
    
    print(f"Found {len(video_files)} video files")
    print(f"Processing videos from: {video_dir}")
    
    for video_file in tqdm(video_files, desc="Extracting FPS"):
        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]  # Bỏ .mp4 extension
        
        fps = get_video_fps(video_path)
        if fps is not None:
            fps_dict[video_name] = fps
            print(f"  {video_name}: {fps} FPS")
        else:
            print(f"  Failed to get FPS for {video_name}")
    
    # Lưu vào file JSON
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(fps_dict, f, indent=4, ensure_ascii=False)
        print(f"\nFPS data saved to: {output_json_path}")
        print(f"Total videos processed: {len(fps_dict)}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")

def extract_fps_from_multiple_directories(base_dir, output_json_path):
    """
    Duyệt qua nhiều thư mục Videos_L01, Videos_L02, etc.
    """
    fps_dict = {}
    
    if not os.path.exists(base_dir):
        print(f"Base directory not found: {base_dir}")
        return
    
    # Tìm tất cả thư mục Videos_*
    video_dirs = []
    for item in os.listdir(base_dir):
        if item.startswith('Videos_L') and os.path.isdir(os.path.join(base_dir, item)):
            video_path = os.path.join(base_dir, item, 'video')
            if os.path.exists(video_path):
                video_dirs.append(video_path)
    
    video_dirs.sort()
    print(f"Found {len(video_dirs)} video directories")
    
    for video_dir in video_dirs:
        print(f"\nProcessing directory: {video_dir}")
        
        if not os.path.exists(video_dir):
            print(f"  Directory not found: {video_dir}")
            continue
        
        # Lấy tất cả file .mp4 trong thư mục
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        video_files.sort()
        
        print(f"  Found {len(video_files)} video files")
        
        for video_file in tqdm(video_files, desc=f"Processing {os.path.basename(video_dir)}"):
            video_path = os.path.join(video_dir, video_file)
            video_name = os.path.splitext(video_file)[0]  # Bỏ .mp4 extension
            
            fps = get_video_fps(video_path)
            if fps is not None:
                fps_dict[video_name] = fps
                # print(f"    {video_name}: {fps} FPS")
            else:
                print(f"    Failed to get FPS for {video_name}")
    
    # Lưu vào file JSON
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(fps_dict, f, indent=4, ensure_ascii=False)
        print(f"\nFPS data saved to: {output_json_path}")
        print(f"Total videos processed: {len(fps_dict)}")
        
        # Hiển thị sample
        print(f"\nSample FPS data:")
        for i, (video_name, fps) in enumerate(fps_dict.items()):
            if i >= 5:  # Chỉ hiển thị 5 video đầu
                break
            print(f"  {video_name}: {fps}")
        if len(fps_dict) > 5:
            print(f"  ... and {len(fps_dict) - 5} more videos")
            
    except Exception as e:
        print(f"Error saving JSON file: {e}")

if __name__ == "__main__":
    # Cấu hình đường dẫn
    base_video_dir = "/dataset/AIC2024/original_dataset/0/videos"
    output_json = "/dataset/AIC_2025/SIU_Sayan/video_fps_0.json"
    
    print("=== VIDEO FPS EXTRACTOR ===")
    print(f"Base directory: {base_video_dir}")
    print(f"Output JSON: {output_json}")
    
    # Kiểm tra nếu chỉ có một thư mục Videos_L01
    single_dir = "/dataset/AIC2024/original_dataset/0/videos/Videos_L01/video"
    if os.path.exists(single_dir):
        print(f"\nProcessing single directory: {single_dir}")
        extract_fps_from_directory(single_dir, output_json)
    else:
        print(f"\nProcessing multiple directories under: {base_video_dir}")
        extract_fps_from_multiple_directories(base_video_dir, output_json) 