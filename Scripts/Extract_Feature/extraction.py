from tqdm import tqdm
import os
import numpy as np
import signal
import sys
from Class.LLM2Clip import LLM2Clip

# Variable to control stopping
stop_processing = False

def signal_handler(sig, frame):
    global stop_processing
    print('\n\nReceived stop signal. Finishing current video processing and stopping safely...')
    stop_processing = True

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

model = LLM2Clip()

main_path = "/dataset/AIC_2025/SIU_Sayan/keyframes"  # Main path
SAVE_DIR = "/dataset/AIC_2025/SIU_Sayan/autoshot/features_llm2clip"
os.makedirs(SAVE_DIR, exist_ok=True)

# Count total and processed videos
total_videos = 0
processed_videos = 0
for level_folder in sorted(os.listdir(main_path)):
    level_folder_path = os.path.join(main_path, level_folder)
    if not os.path.isdir(level_folder_path) or level_folder == "SceneJson":
        continue
    keyframes_path = os.path.join(level_folder_path, "keyframes")
    if os.path.exists(keyframes_path):
        total_videos += len([d for d in os.listdir(keyframes_path) if os.path.isdir(os.path.join(keyframes_path, d))])

print(f"Total videos to process: {total_videos}")

# Iterate through level folders (Keyframes_L21, Keyframes_L22, ...)
for level_folder in tqdm(sorted(os.listdir(main_path))):
    if stop_processing:
        print("Processing stopped as requested.")
        break
        
    level_folder_path = os.path.join(main_path, level_folder)
    if not os.path.isdir(level_folder_path) or level_folder == "SceneJson":
        continue
    
    print(f"Processing level: {level_folder}")
    
    # Enter the keyframes subfolder
    keyframes_path = os.path.join(level_folder_path, "keyframes")
    if not os.path.exists(keyframes_path):
        print(f"Keyframes folder not found in {level_folder}")
        continue
    
    # Iterate through video folders within keyframes (L21_V001, L21_V002, ...)
    for video_folder in tqdm(sorted(os.listdir(keyframes_path)), desc=f"Processing {level_folder}"):
        if stop_processing:
            print("Processing stopped as requested.")
            break
            
        video_folder_path = os.path.join(keyframes_path, video_folder)
        if not os.path.isdir(video_folder_path):
            continue
        
        # Check if file has already been processed
        save_name = f"{video_folder}.npy"
        save_path = os.path.join(SAVE_DIR, save_name)
        
        if os.path.exists(save_path):
            processed_videos += 1
            print(f"Skipped {video_folder} (already processed). Progress: {processed_videos}/{total_videos}")
            continue
        
        result = []
        # Iterate through image files in the video folder
        for keyframe in tqdm(sorted(os.listdir(video_folder_path)), desc=f"Processing {video_folder}", leave=False):
            if stop_processing:
                print("Processing stopped as requested.")
                break
                
            if not keyframe.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue
            keyframe_path = os.path.join(video_folder_path, keyframe)
            try:
                feet = model.get_image_features(keyframe_path)
                result.append(feet)
            except Exception as e:
                print(f"Error processing {keyframe_path}: {e}")
                continue
        
        if stop_processing:
            break
            
        # Save file by video folder name, e.g., L01_V001.npy
        if result:  # Only save if we have features
            np.save(save_path, result)
            processed_videos += 1
            print(f"Saved {video_folder}: {len(result)} features. Progress: {processed_videos}/{total_videos}")
        else:
            print(f"No features extracted for {video_folder}")

if stop_processing:
    print(f"\nProcessing stopped. Completed {processed_videos}/{total_videos} videos.")
else:
    print(f"\nAll done! Processed {processed_videos}/{total_videos} videos.")
