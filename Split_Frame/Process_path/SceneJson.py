import os
import shutil

# Thư mục gốc chứa tất cả các thư mục Videos_L01, Videos_L02, ...
base_dir = '/dataset/AIC_2025/SIU_Sayan/autoshot/SceneJson'  # 🔁 Đổi thành đường dẫn thực tế trên máy bạn

# Duyệt qua từng thư mục con
for i in range(1, 13):
    folder_name = f'Videos_L{str(i).zfill(2)}'
    video_folder_path = os.path.join(base_dir, folder_name, 'video')
    
    if os.path.exists(video_folder_path):
        print(f"Đang xử lý: {video_folder_path}")
        
        # Di chuyển tất cả file .json ra thư mục cha
        for filename in os.listdir(video_folder_path):
            if filename.endswith('.json'):
                src_file = os.path.join(video_folder_path, filename)
                dest_file = os.path.join(base_dir, folder_name, filename)
                shutil.move(src_file, dest_file)
                print(f"  → Đã di chuyển: {filename}")
        
        # Xoá thư mục 'video' sau khi di chuyển xong
        shutil.rmtree(video_folder_path)
        print(f"  ✂️ Đã xoá thư mục: {video_folder_path}\n")
    else:
        print(f"❌ Không tìm thấy: {video_folder_path}")
