import os
import shutil

# Đường dẫn gốc chứa các thư mục Videos_L01, ..., Videos_L12
base_path = '/dataset/AIC_2025/SIU_Sayan/autoshot/keyframess'  # 🔁 Thay bằng đường dẫn thực tế

# Lặp qua từng thư mục Videos_L01 → Videos_L12
for i in range(1, 13):
    folder_name = f'Videos_L{str(i).zfill(2)}'
    folder_path = os.path.join(base_path, folder_name)
    video_path = os.path.join(folder_path, 'video')

    if os.path.exists(video_path):
        print(f"📂 Đang xử lý: {video_path}")
        subdirs = [d for d in os.listdir(video_path) if os.path.isdir(os.path.join(video_path, d))]
        
        for subdir in subdirs:
            src = os.path.join(video_path, subdir)
            dst = os.path.join(folder_path, subdir)
            if not os.path.exists(dst):
                shutil.move(src, dst)
                print(f"  ✅ Di chuyển: {subdir}")
            else:
                print(f"  ⚠️ Bỏ qua {subdir} (đã tồn tại)")

        # Xoá thư mục video rỗng
        shutil.rmtree(video_path)
        print(f"  🗑️ Đã xoá thư mục: {video_path}\n")
    else:
        print(f"❌ Không tìm thấy: {video_path}")
