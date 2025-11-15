import cv2

def get_video_fps(video_path):
    # Mở video
    cap = cv2.VideoCapture(video_path)
    
    # Kiểm tra nếu video mở thành công
    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
        return None
    
    # Lấy FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Đóng video
    cap.release()
    
    return fps

# Ví dụ sử dụng
video_path = "/dataset/AIC_2025/SIU_Sayan/video/Videos_L24/L24_V044.mp4"
fps = get_video_fps(video_path)
if fps is not None:
    print(f"FPS của video là: {fps}")