# Hệ Thống Tìm Kiếm Video AIC 2025 - SIU Sayan

Hệ thống tìm kiếm video thông minh sử dụng nhiều mô hình CLIP và FAISS GPU để tìm kiếm video/frame dựa trên text hoặc image query. Hệ thống hỗ trợ đầy đủ các tính năng tìm kiếm, OCR, ASR và submission cho cuộc thi AIC 2025.

## 📋 Mục Lục

- [Tổng Quan](#tổng-quan)
- [Tính Năng](#tính-năng)
- [Kiến Trúc Hệ Thống](#kiến-trúc-hệ-thống)
- [Cài Đặt](#cài-đặt)
- [Cấu Hình](#cấu-hình)
- [Sử Dụng](#sử-dụng)
- [API Endpoints](#api-endpoints)
- [Submission Types](#submission-types)
- [Cấu Trúc Thư Mục](#cấu-trúc-thư-mục)
- [Troubleshooting](#troubleshooting)

## 🎯 Tổng Quan

Hệ thống này được phát triển cho cuộc thi **AIC 2025 (AI Challenge)** với các tính năng chính:

- **Tìm kiếm video/frame** bằng text hoặc image query
- **Hỗ trợ nhiều mô hình CLIP** khác nhau để tối ưu độ chính xác
- **FAISS GPU** để tăng tốc độ tìm kiếm vector
- **Web interface** trực quan và dễ sử dụng
- **OCR tiếng Việt** để nhận dạng text trong video
- **ASR** để chuyển đổi audio thành text
- **Shot detection** để phân đoạn video tự động
- **Submission** cho KIS, QA, và TRAKE tasks

## ✨ Tính Năng

### 1. Tìm Kiếm Đa Mô Hình
Hệ thống hỗ trợ 8 mô hình CLIP khác nhau:
- **DFN5B** - Apple DFN5B-CLIP-ViT-H-14
- **LAION** - LAION CLIP
- **MetaCLIP** - Meta CLIP H14
- **MetaCLIP2** - Meta CLIP2 Worldwide Huge
- **SigLIP** - Google SigLIP SO400M
- **SigLIP2** - Google SigLIP2 Giant
- **JinaCLIPV2** - Jina AI CLIP V2
- **LLM2Clip** - Microsoft LLM2CLIP

### 2. Tìm Kiếm Vector với FAISS GPU
- Sử dụng FAISS GPU để tăng tốc độ tìm kiếm
- Hỗ trợ nhiều loại index: `HIGH_ACCURACY`, `HIGH_ACCURACY_FP16`
- Tìm kiếm real-time với độ trễ thấp

### 3. Web Interface
- Giao diện web trực quan
- Tìm kiếm bằng text hoặc image
- Xem video và frame kết quả
- Hỗ trợ temporal search và RRF (Reciprocal Rank Fusion)

### 4. OCR & ASR
- **OCR tiếng Việt** với PaddleOCR và Vietnamese OCR
- **ASR** để chuyển đổi audio thành text
- Hỗ trợ nhận dạng text trong video frames

### 5. Shot Detection
- Tự động phân đoạn video thành các shots
- Sắp xếp kết quả theo shot grouping
- Hỗ trợ file JSON chứa thông tin shots

### 6. Submission System
- **KIS (Known-Item Search)**: Tìm kiếm video/frame cụ thể
- **QA (Question Answering)**: Trả lời câu hỏi về video
- **TRAKE (Temporal Ranking)**: Xếp hạng nhiều frames theo thời gian

## 🏗️ Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────┐
│                    Web Interface                         │
│              (FastAPI + Jinja2 Templates)                │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
┌────────▼────────┐    ┌─────────▼─────────┐
│  API Endpoints  │    │  Combine API      │
│  (8 models)     │    │  (Temporal/RRF)   │
└────────┬────────┘    └─────────┬─────────┘
         │                       │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   Feature Extractors  │
         │  (CLIP Models)        │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   FAISS GPU Database  │
         │   (Vector Search)     │
         └───────────────────────┘
```

## 📦 Cài Đặt

### Yêu Cầu Hệ Thống
- Python 3.8+
- CUDA-capable GPU (khuyến nghị)
- RAM: 16GB+ (khuyến nghị 32GB)
- Disk: 100GB+ cho models và cache

### Cài Đặt Dependencies

```bash
# Clone repository
git clone <repository-url>
cd Base

# Cài đặt dependencies cho FAISS
pip install -r Scripts/requirements_faiss.txt

# Cài đặt dependencies cho OCR (nếu cần)
pip install -r Scripts/OCR/requirements.txt

# Cài đặt các dependencies khác
pip install fastapi uvicorn torch torchvision transformers
pip install open-clip-torch pillow numpy ujson
pip install faiss-gpu  # hoặc faiss-cpu nếu không có GPU
```

### Tải Models

Models sẽ được tự động tải về khi chạy lần đầu tiên. Tất cả models được lưu trong thư mục `Cache/`.

**Lưu ý**: Models có thể rất lớn (vài GB mỗi model), đảm bảo có đủ dung lượng disk.

## ⚙️ Cấu Hình

### Environment Variables

Tạo file `.env` hoặc set các biến môi trường:

```bash
export HF_HOME=/workspace/competitions/AIC_2025/SIU_Sayan/Base/Cache
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

### Cấu Hình Paths

Các paths quan trọng trong hệ thống:

- **Cache directory**: `/workspace/competitions/AIC_2025/SIU_Sayan/Base/Cache`
- **Features directory**: `/dataset/AIC_2025/SIU_Sayan/autoshot/features_*`
- **Keyframes directory**: `/dataset/AIC_2025/SIU_Sayan/keyframes`
- **Scene JSON**: `/dataset/AIC_2025/SIU_Sayan/autoshot/SceneJson`

### Cấu Hình API Ports

Các API endpoints mặc định:

| Model | Port |
|-------|------|
| DFN5B | 8501 |
| LAION | 8502 |
| SigLIP | 8503 |
| Combine | 8504 |
| JinaCLIPV2 | 8505 |
| Translate | 8506 |
| MetaCLIP2 | 8510 |
| MetaCLIP | 8511 |
| LLM2Clip | 8512 |
| SigLIP2 | 8513 |
| Web Server | 8000 |

## 🚀 Sử Dụng

### 1. Khởi Động API Servers

Khởi động từng API server cho mỗi model:

```bash
cd Scripts/API

# DFN5B API
python API_DFN5B_FAISS.py

# LAION API
python API_Laion_FAISS.py

# SigLIP API
python API_SigLIP_FAISS.py

# ... và các API khác
```

Hoặc sử dụng script để khởi động tất cả:

```bash
# Tạo script khởi động (cần tự tạo)
./start_all_apis.sh
```

### 2. Khởi Động Web Server

```bash
cd Scripts/Web
python server.py
```

Web interface sẽ có sẵn tại: `http://localhost:8000`

### 3. Sử Dụng Web Interface

1. Mở trình duyệt và truy cập `http://localhost:8000`
2. Chọn model từ dropdown
3. Nhập text query hoặc upload image
4. Click "Search" để tìm kiếm
5. Xem kết quả và click vào frame để xem chi tiết

### 4. Sử Dụng API Trực Tiếp

#### Text Search

```bash
curl -X POST "http://localhost:8501/text_search" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "a person walking",
    "k": 200
  }'
```

#### Image Search

```bash
curl -X POST "http://localhost:8501/image_search" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "k": 200
  }'
```

## 📡 API Endpoints

### Text Search
```
POST /text_search
Body: {
  "text": "query text",
  "k": 200
}
```

### Image Search
```
POST /image_search
Body: {
  "image_url": "url or path",
  "k": 200
}
```

### Preprocess
```
POST /preprocess
Body: {
  "text": "text to preprocess"
}
```

### Health Check
```
GET /health
```

### Combine APIs

#### Temporal Search
```
POST /search_temporal
Body: {
  "text": "query",
  "k": 200,
  "models": ["siglip", "dfn5b", ...]
}
```

#### RRF Search
```
POST /search_rrf
Body: {
  "text": "query",
  "k": 200,
  "models": ["siglip", "dfn5b", ...]
}
```

## 📝 Submission Types

### 1. KIS (Known-Item Search)

Tìm kiếm video/frame cụ thể:

```json
{
  "answerSets": [{
    "answers": [{
      "mediaItemName": "K03_V019",
      "start": 399333,
      "end": 399333
    }]
  }]
}
```

**Cách sử dụng trong Web Interface:**
1. Tìm kiếm video/frame
2. Click vào kết quả → auto-fill video name và frame
3. Click "Nộp bài" → tự động submit KIS

### 2. QA (Question Answering)

Trả lời câu hỏi về video:

```json
{
  "answerSets": [{
    "answers": [{
      "text": "QA-12345-K03_V019-399333"
    }]
  }]
}
```

**Cách sử dụng:**
1. Tìm kiếm video/frame
2. Click vào kết quả
3. **Nhập answer vào field "QA Answer"**
4. Click "Nộp bài" → tự động submit QA

### 3. TRAKE (Temporal Ranking)

Xếp hạng nhiều frames theo thời gian:

```json
{
  "answerSets": [{
    "answers": [{
      "text": "TR-K03_V019-11980,12000,12050"
    }]
  }]
}
```

**Cách sử dụng:**
1. Click nút ▶️ Play để mở video modal
2. Nhập 2-4 frame IDs vào "Add Frames"
3. Click **"Submit TRAKE"** (button màu xanh lá)

Xem thêm chi tiết trong file `Scripts/DRES_SUBMISSION_GUIDE.md`

## 📁 Cấu Trúc Thư Mục

```
Base/
├── Cache/                    # Models và cache files
│   ├── hub/                 # HuggingFace cache
│   ├── models--*/           # Các models đã tải
│   └── faiss_indices/       # FAISS indices
├── Scripts/
│   ├── API/                 # API endpoints cho các models
│   │   ├── API_DFN5B_FAISS.py
│   │   ├── API_Laion_FAISS.py
│   │   ├── API_SigLIP_FAISS.py
│   │   ├── API_LLM2Clip_FAISS.py
│   │   ├── Combine.py       # Combine multiple models
│   │   └── Translate.py     # Translation API
│   ├── Extract_Feature/     # Feature extraction scripts
│   │   ├── Class/           # Model classes
│   │   └── *_extraction.py  # Extraction scripts
│   ├── Vector_database/     # FAISS database
│   │   ├── faiss_gpu.py
│   │   └── faiss_gpu_llm2clip.py
│   ├── Web/                 # Web interface
│   │   ├── server.py
│   │   └── templates/
│   ├── OCR/                 # OCR modules
│   ├── ASR/                 # ASR modules
│   └── Utils/               # Utility functions
├── Split_Frame/             # Shot detection
│   └── Shot_Detection-main/
├── .gitignore
└── README.md
```

## 🔧 Troubleshooting

### Lỗi GPU không được sử dụng

```bash
# Kiểm tra CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Kiểm tra FAISS GPU
python -c "import faiss; print(faiss.get_num_gpus())"
```

### Lỗi Model không tải được

- Kiểm tra kết nối internet
- Kiểm tra dung lượng disk
- Kiểm tra path `HF_HOME` trong environment variables

### Lỗi FAISS Index không tìm thấy

- Đảm bảo features đã được extract
- Kiểm tra path đến features directory
- Chạy lại script extraction nếu cần

### Lỗi Port đã được sử dụng

```bash
# Tìm process đang sử dụng port
lsof -i :8501

# Kill process
kill -9 <PID>
```

### Memory Issues

- Giảm số lượng models chạy đồng thời
- Sử dụng `device="cpu"` cho một số models
- Giảm batch size trong extraction

## 📚 Tài Liệu Tham Khảo

- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Open CLIP](https://github.com/mlfoundations/open_clip)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

## 📄 License

[Thêm thông tin license nếu có]

## 👥 Contributors

SIU Sayan Team - AIC 2025

---

**Lưu ý**: Đây là hệ thống được phát triển cho cuộc thi AIC 2025. Một số paths và cấu hình có thể cần điều chỉnh tùy theo môi trường triển khai.

