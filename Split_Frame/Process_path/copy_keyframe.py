import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Đường dẫn gốc và đích
src_root = '/dataset/AIC2024/pumkin_dataset/0/utils/autoshot/keyframes'
dst_root = '/dataset/AIC_2025/SIU_Sayan/autoshot/keyframes'

# Tạo thư mục đích nếu chưa có
os.makedirs(dst_root, exist_ok=True)

# Lấy danh sách tất cả thư mục con trong thư mục nguồn
subdirs = [name for name in os.listdir(src_root)
           if os.path.isdir(os.path.join(src_root, name))]

def copy_folder(subdir_name):
    src_path = os.path.join(src_root, subdir_name)
    dst_path = os.path.join(dst_root, subdir_name)
    try:
        shutil.copytree(src_path, dst_path)
        return f"✅ Đã copy {subdir_name}"
    except Exception as e:
        return f"❌ Lỗi khi copy {subdir_name}: {e}"

# Copy song song bằng nhiều luồng
with ThreadPoolExecutor(max_workers=64) as executor:
    futures = [executor.submit(copy_folder, name) for name in subdirs]
    for future in as_completed(futures):
        print(future.result())
