from fastapi import FastAPI, Request, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import aiohttp
import asyncio
import time
import json
import os
import sys
from typing import Optional
import requests
import subprocess
import shutil
from datetime import datetime
import zipfile

# Environment setup
os.environ['HF_HOME'] = '/workspace/competitions/AIC_2025/SIU_Sayan/Base/Cache'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sys.path.append("/workspace/competitions/AIC_2025/SIU_Sayan/Base/Scripts/")

app = FastAPI(
    title="AIC 2025 SIU Sayan Web Interface - FAISS GPU",
    description="Web interface for video search using multiple FAISS GPU AI models",
    version="1.0.0",
    root_path="/siu_sayan_9"
)

# Setup templates - Fixed to use absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(script_dir, "templates")
templates = Jinja2Templates(directory=template_dir)

# API endpoints - Updated for FAISS GPU
API_ENDPOINTS = {
    "siglip": "http://localhost:8503/text_search",
    "laion": "http://localhost:8502/text_search",
    "dfn5b": "http://localhost:8501/text_search",
    "jinaclipv2": "http://localhost:8505/text_search",
    "metaclip": "http://localhost:8511/text_search",
    "metaclip2": "http://localhost:8510/text_search",
    "siglip2": "http://localhost:8513/text_search",
    "llm2clip": "http://localhost:8512/text_search",
    "combine_temporal": "http://localhost:8504/search_temporal",
    "combine_rrf": "http://localhost:8504/search_rrf",
    # "reform": "http://localhost:8505/reform",  # REMOVED - No longer needed
    "translate": "http://localhost:8506/translate"
}

# Dataset paths - Updated for AIC 2025
KEYFRAME_FOLDER_PATH = "/dataset/AIC_2025/SIU_Sayan/keyframes_webp"
VIDEOS_FOLDER_PATH = "/dataset/AIC_2025/SIU_Sayan/autoshot/videos"

# Load FPS results
FPS_FILE_PATH = "/workspace/competitions/AIC_2025/SIU_Sayan/Base/Scripts/Web/fps_results.json"
fps_results = {}

# Try to load FPS results, create default if not exists
try:
    if os.path.exists(FPS_FILE_PATH):
        with open(FPS_FILE_PATH, "r") as f:
            fps_results = json.load(f)
    else:
        # Create default FPS file
        fps_results = {}
        os.makedirs(os.path.dirname(FPS_FILE_PATH), exist_ok=True)
        with open(FPS_FILE_PATH, "w") as f:
            json.dump(fps_results, f)
except Exception as e:
    print(f"Warning: Could not load FPS results: {e}")
    fps_results = {}

VIDEO_FPS_PATH = "/dataset/AIC_2025/SIU_Sayan/autoshot/video_fps_0.json"
video_fps = {}
try:
    if os.path.exists(VIDEO_FPS_PATH):
        with open(VIDEO_FPS_PATH, "r", encoding="utf-8-sig") as f:
            video_fps = json.load(f)
    else:
        print(f"Warning: FPS file not found at {VIDEO_FPS_PATH}")
except Exception as e:
    print(f"Warning: Could not load video_fps_0.json: {e}")
    video_fps = {}

DRES_BASE_URL = "https://eventretrieval.oj.io.vn"
DRES_USERNAME = "team035"
DRES_PASSWORD = "t3XzzxQZPd"

def get_session_id():
    login_resp = requests.post(
        f"{DRES_BASE_URL}/api/v2/login",
        json={"username": DRES_USERNAME, "password": DRES_PASSWORD}
    )
    if login_resp.status_code == 200:
        return login_resp.json()["sessionId"]
    return None

@app.get("/img/{filename:path}")
async def download_file(filename: str):
    """Serve keyframe images"""
    try:
        print(f"Request for image: {filename}")
        
        # Xử lý đường dẫn đặc biệt cho cấu trúc Keyframes_K01/keyframes/K01_V001/7713.webp
        parts = filename.split("/")
        
        # Nếu có cấu trúc Keyframes_XXX/keyframes/XXX_VXXX/frame.webp (4 phần)
        if len(parts) >= 4:
            keyframes_folder = parts[-4]  # Keyframes_K01
            keyframes_subfolder = parts[-3]  # keyframes
            video_name = parts[-2]    # K01_V001
            frame_name = parts[-1]    # 7713.webp
            
            # Tạo đường dẫn đầy đủ
            full_path = os.path.join(KEYFRAME_FOLDER_PATH, keyframes_folder, keyframes_subfolder, video_name, frame_name)
        # Fallback cho cấu trúc cũ hoặc 3 phần
        elif len(parts) >= 3:
            video_folder = parts[-3]  # Keyframes_K01
            video_name = parts[-2]    # K01_V001
            frame_name = parts[-1]    # 7713.webp
            
            # Tạo đường dẫn đầy đủ (thêm keyframes subfolder)
            full_path = os.path.join(KEYFRAME_FOLDER_PATH, video_folder, "keyframes", video_name, frame_name)
            print(f"Trying path with 3 parts: {full_path}")
            
            if os.path.exists(full_path):
                print(f"Found image at: {full_path}")
                return FileResponse(full_path)
        
        # Nếu có cấu trúc XXX_VXXX/frame.webp (2 phần)
        if len(parts) >= 2:
            video_name = parts[-2]
            frame_name = parts[-1]
            
            # Tìm thư mục Keyframes_XXX dựa trên XXX_VXXX
            batch_part = video_name[:3]  # e.g., "K01"
            keyframes_folder = f"Keyframes_{batch_part}"
            
            full_path = os.path.join(KEYFRAME_FOLDER_PATH, keyframes_folder, "keyframes", video_name, frame_name)
            print(f"Trying path with 2 parts: {full_path}")
            
            if os.path.exists(full_path):
                print(f"Found image at: {full_path}")
                return FileResponse(full_path)
        
        # Thử đường dẫn trực tiếp
        direct_path = os.path.join(KEYFRAME_FOLDER_PATH, filename)
        print(f"Trying direct path: {direct_path}")
        
        if os.path.exists(direct_path):
            print(f"Found image at: {direct_path}")
            return FileResponse(direct_path)
            
        print(f"Image not found: {filename}")
        print(f"KEYFRAME_FOLDER_PATH: {KEYFRAME_FOLDER_PATH}")
            
        raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        print(f"Error serving image {filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error serving image: {str(e)}")

@app.get("/video/{filename:path}")
async def download_video(filename: str):
    """Serve video files"""
    try:
        # Handle different filename formats
        if filename.endswith('.mp4'):
            # Remove .mp4 extension to get the base name
            base_name = filename.replace('.mp4', '')
            
            # Extract batch info (L21, L22, L23, etc.)
            batch_part = base_name[:3]  # e.g., "L21"
            
            # Construct the correct folder structure: Videos_L21/video/L21_V001.mp4
            video_folder = f"Videos_{batch_part}"
            full_path = os.path.join(VIDEOS_FOLDER_PATH, video_folder, "video", filename)
        else:
            # Direct path
            full_path = os.path.join(VIDEOS_FOLDER_PATH, filename)
        
        if os.path.exists(full_path):
            return FileResponse(full_path)
        
        # Log for debugging
        print(f"Video not found at: {full_path}")
        raise HTTPException(status_code=404, detail=f"Video not found: {filename}")
    except Exception as e:
        print(f"Error serving video {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error serving video: {str(e)}")

async def call_api(url: str, params: dict = None) -> dict:
    """Helper function to call APIs asynchronously"""
    try:
        print(f"📡 DEBUG: Calling API - URL: {url}")
        print(f"📡 DEBUG: Parameters: {params}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=30) as response:
                print(f"📡 DEBUG: API Response Status: {response.status}")
                if response.status == 200:
                    result = await response.json()
                    if isinstance(result, list):
                        print(f"📡 DEBUG: API returned {len(result)} results")
                    elif isinstance(result, dict) and "results" in result:
                        print(f"📡 DEBUG: API returned {len(result['results'])} results")
                    else:
                        print(f"📡 DEBUG: API returned unexpected format: {type(result)}")
                    return result
                else:
                    print(f"API error {response.status}: {url}")
                    return None
    except Exception as e:
        print(f"Error calling API: {e}")
        return None

async def call_api_post(url: str, data: dict) -> dict:
    """Helper function to call APIs with POST"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, timeout=30) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"API error {response.status}: {url}")
                    return None
    except Exception as e:
        print(f"Error calling API: {e}")
        return None

def format_results(api_results: list, model_type: str = "single") -> list:
    """Format API results for display"""
    files = []
    
    if not api_results:
        print("⚠️ DEBUG: No API results received")
        return files
    
    print(f"🔍 DEBUG: Formatting {len(api_results)} API results (model_type: {model_type})")
    
    try:
        # Handle different result formats
        if model_type == "combine":
            # Results from combine API: [[video_name, keyframe_id], ...]
            for i, result in enumerate(api_results):
                if len(result) >= 2:
                    video_name = result[0]
                    keyframe_id = result[1]
                    before_count = len(files)
                    process_single_result(files, video_name, keyframe_id)
                    after_count = len(files)
                    if before_count == after_count:
                        print(f"⚠️ DEBUG: Result {i} was skipped: {video_name}, {keyframe_id}")
                else:
                    print(f"⚠️ DEBUG: Invalid combine result format at index {i}: {result}")
        else:
            # Results from single model APIs
            for i, result in enumerate(api_results):
                if isinstance(result, dict):
                    video_name = result.get('video_name', '')
                    keyframe_id = result.get('keyframe_id', '')
                    if video_name and keyframe_id:
                        before_count = len(files)
                        process_single_result(files, video_name, keyframe_id)
                        after_count = len(files)
                        if before_count == after_count:
                            print(f"⚠️ DEBUG: Result {i} was skipped: {video_name}, {keyframe_id}")
                    else:
                        print(f"⚠️ DEBUG: Missing video_name or keyframe_id at index {i}: {result}")
                else:
                    print(f"⚠️ DEBUG: Invalid single result format at index {i}: {result}")
                        
    except Exception as e:
        print(f"❌ Error formatting results: {e}")
    
    print(f"✅ DEBUG: Final formatted results count: {len(files)}/{len(api_results)}")
    return files

def process_single_result(files: list, video_name: str, keyframe_id: str):
    """Process a single result and add to files list"""
    try:
        # Remove .mp4 extension if present
        folder_name = video_name.replace('.mp4', '')
        
        # Extract batch info (L21, L22, L23, etc.)
        batch_part = folder_name[:3]  # e.g., "L21"
        
        # Construct the correct folder structure: Keyframes_L21/keyframes/L21_V001
        keyframes_folder = f"Keyframes_{batch_part}"
        full_folder_path = f"{keyframes_folder}/keyframes/{folder_name}"
        
        # Keep keyframe_id format with leading zeros to match actual filenames (00000.webp, 00026.webp)
        keyframe_filename = keyframe_id.zfill(5)  # Ensure 5-digit format with leading zeros
        
        # REMOVED: Check if file exists before adding to results - để đảm bảo luôn trả về đủ 200 kết quả
        # actual_file_path = f"{KEYFRAME_FOLDER_PATH}{full_folder_path}/{keyframe_filename}.webp"
        # if not os.path.exists(actual_file_path):
        #     print(f"Warning: Image not found at {actual_file_path}")
        #     return
        
        # Create web-accessible path (used by frontend) - use filename without leading zeros
        web_accessible_path = f"/img/{full_folder_path}/{keyframe_filename}.webp"
        
        # Get FPS from video_fps_0.json (default to 25 if not found)
        # Try without .mp4 extension first, then with .mp4 extension
        fps = video_fps.get(folder_name, video_fps.get(f"{folder_name}.mp4", 25))
        
        # Calculate time using the original keyframe_id (preserve original for time calculation)
        true_key_frame = int(keyframe_id)
        keyframe_time = true_key_frame / fps
        
        # Format time - use proper rounding instead of truncation
        minutes = int(keyframe_time // 60)
        seconds = round(keyframe_time % 60)  # Round seconds instead of truncating
        time_str = f"{minutes:02d}:{seconds:02d}"
        
        # Create display ID using filename with leading zeros
        right_id = f"{time_str},{keyframes_folder},{folder_name},{keyframe_filename}"
        
        # Store web-accessible path instead of full system path
        files.append((web_accessible_path, right_id, int(keyframe_time)))
        
    except Exception as e:
        print(f"❌ ERROR: Error processing result {video_name}, {keyframe_id}: {e}")
        # Vẫn có thể add một kết quả mặc định để không mất số lượng
        try:
            web_accessible_path = f"/img/error/{video_name}_{keyframe_id}.webp"
            files.append((web_accessible_path, f"error,{video_name},{keyframe_id}", 0))
        except:
            pass

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page"""
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "files": [],
        "query_text": "",
        "selected_model": "laion",  # Default model
        "trans_checked": False,
        "temporal_checked": False,
        "k_value": 200
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    text: str = Form(...),
    model: str = Form("laion"),
    trans: int = Form(0),
    temporal: int = Form(0),
    k: int = Form(200),
    results_limit: int = Form(200)
):
    """Handle search predictions"""
    start_time = time.time()
    files = []
    
    # Use results_limit if it's different from default, otherwise use k
    # This allows custom result limit to work properly
    if results_limit != 200:
        actual_limit = results_limit
        print(f"🔍 DEBUG: Using custom results_limit: {actual_limit}")
    else:
        actual_limit = k
        print(f"🔍 DEBUG: Using default k: {actual_limit}")
    
    print(f"🔍 DEBUG: Final limit parameters - results_limit={results_limit}, k={k}, actual_limit={actual_limit}")
    
    # Preserve original text for display
    original_query_text = text
    
    try:
        # Handle translation if needed
        if trans != 0:
            print("Translating Vietnamese text to English...")
            translation_result = await call_api_post(
                API_ENDPOINTS["translate"], 
                {"text": text, "source": "vi", "target": "en"}
            )
            if translation_result and "translated_text" in translation_result:
                original_text = text
                text = translation_result["translated_text"]
                print(f"Original text (VI): {original_text}")
                print(f"Translated text (EN): {text}")
            else:
                print("Translation failed, using original text")
        
        # Apply text reform/summarization to English text
        # This happens after translation or with original English text
        # REFORM FUNCTIONALITY REMOVED - No longer needed
        # try:
        #     print("Applying text reform/summarization...")
        #     reform_result = await call_api_post(
        #         API_ENDPOINTS["reform"],
        #         {"text": text, "max_length": 100}
        #     )
        #     if reform_result and "reformed_text" in reform_result:
        #         text = reform_result["reformed_text"]
        #         print(f"Reformed text: {text}")
        #     else:
        #         print("Reform failed, using translated/original text")
        # except Exception as e:
        #     print(f"Reform error: {e}, using translated/original text")
        
        # Check if input is an image URL/path for image search
        is_image_search = False
        if text and (text.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')) or 
                    '/img/' in text.lower() or 
                    text.lower().startswith(('http', '/dataset', '/workspace'))):
            is_image_search = True
            print(f"🖼️ Detected image URL/path: {text}")
        
        # Prepare search parameters
        if is_image_search:
            search_params = {
                "image_url": text,
                "k": actual_limit
            }
        else:
            search_params = {
                "text": text,
                "k": actual_limit
            }
        
        # Handle different search types
        if temporal != 0 and not is_image_search:
            # Temporal search (only for text, not image)
            print("Performing temporal search...")
            search_params["model"] = model
            result = await call_api(API_ENDPOINTS["combine_temporal"], search_params)
            
            if result and "results" in result:
                print(f"🔍 DEBUG: Temporal API returned {len(result['results'])} results")
                files = format_results(result["results"], "combine")
            else:
                print("⚠️ DEBUG: Temporal API returned no results or invalid format")
                
        elif model == "Combine" and not is_image_search:
            # RRF combine search (only for text, not image)
            print("Performing RRF combine search...")
            search_params["models"] = "['siglip','laion','dfn5b','jinaclipv2','metaclip','metaclip2','siglip2','llm2clip']"
            result = await call_api(API_ENDPOINTS["combine_rrf"], search_params)
            
            if result and "results" in result:
                print(f"🔍 DEBUG: RRF Combine API returned {len(result['results'])} results")
                files = format_results(result["results"], "combine")
            else:
                print("⚠️ DEBUG: RRF Combine API returned no results or invalid format")
                
        else:
            # Single model search
            if is_image_search:
                print(f"Performing {model} IMAGE search...")
                if model.lower() in API_ENDPOINTS:
                    # Replace /text_search with /image_search for image queries
                    api_url = API_ENDPOINTS[model.lower()].replace("/text_search", "/image_search")
                    result = await call_api(api_url, search_params)
                else:
                    print(f"Model {model} not supported for image search")
                    result = None
            else:
                print(f"Performing {model} TEXT search...")
                if model.lower() in API_ENDPOINTS:
                    api_url = API_ENDPOINTS[model.lower()]
                    result = await call_api(api_url, search_params)
                else:
                    print(f"Unknown model: {model}")
                    result = None
            
            # Process results (for both image and text search)
            if result:
                if isinstance(result, list):
                    print(f"🔍 DEBUG: {model} API returned {len(result)} results (list format)")
                    files = format_results(result, "single")
                elif isinstance(result, dict) and "results" in result:
                    print(f"🔍 DEBUG: {model} API returned {len(result['results'])} results (dict format)")
                    files = format_results(result["results"], "single")
                else:
                    print(f"⚠️ DEBUG: {model} API returned unexpected format: {type(result)}")
                    # Try to handle as direct list
                    files = format_results(result, "single")
            else:
                print(f"⚠️ DEBUG: {model} API returned None")
        
        processing_time = time.time() - start_time
        print(f"Total processing time: {processing_time:.2f}s")
        print(f"🎯 FINAL RESULT: Found {len(files)} results out of requested {actual_limit}")
        
    except Exception as e:
        print(f"Error in predict: {e}")
        import traceback
        traceback.print_exc()
        files = []
    
    # Debug: Print values being passed to template
    print(f"🐛 DEBUG - Template values:")
    print(f"   original_query_text: '{original_query_text}'")
    print(f"   processed_text: '{text}'")
    print(f"   selected_model: {model}")
    print(f"   trans_checked: {trans}")
    print(f"   temporal_checked: {temporal}")
    print(f"   is_image_search: {is_image_search if 'is_image_search' in locals() else False}")
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "files": files,
        "query_text": original_query_text,
        "processing_time": f"{processing_time:.2f}s" if 'processing_time' in locals() else "N/A",
        "selected_model": model,
        "trans_checked": trans,
        "temporal_checked": temporal,
        "k_value": actual_limit
    })

@app.get("/predict", response_class=HTMLResponse)
async def predict_get(request: Request):
    """Handle GET requests to predict endpoint"""
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "files": [],
        "query_text": "",
        "selected_model": "laion",
        "trans_checked": False,
        "temporal_checked": False,
        "k_value": 200
    })

@app.get("/test_template")
async def test_template(request: Request, model: str = "siglip"):
    """Test endpoint to verify template variables"""
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "files": [],
        "query_text": "Test query",
        "selected_model": model,
        "trans_checked": 1,
        "temporal_checked": 0,
        "k_value": 200
    })

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "api_endpoints": API_ENDPOINTS,
        "keyframe_path": KEYFRAME_FOLDER_PATH,
        "videos_path": VIDEOS_FOLDER_PATH
    }

@app.get("/api/test")
async def test_apis():
    """Test all API endpoints"""
    results = {}
    
    test_params = {
        "text": "test query",
        "k": 5,
        "video_filter": "",
        "time_in": "",
        "time_out": "",
        "batchs": "[0,1,2]"
    }
    
    for name, url in API_ENDPOINTS.items():
        try:
            if "combine" in name:
                if "temporal" in name:
                    test_params["model"] = "siglip"
                result = await call_api(url, test_params)
            else:
                result = await call_api(url, test_params)
            
            results[name] = {
                "status": "ok" if result else "error",
                "url": url,
                "response_type": type(result).__name__
            }
        except Exception as e:
            results[name] = {
                "status": "error",
                "url": url,
                "error": str(e)
            }
    
    return results

@app.get("/api/video_fps")
async def get_video_fps(video_name: str):
    """Get FPS for a specific video"""
    try:
        # Remove .mp4 extension if present
        folder_name = video_name.replace('.mp4', '')
        
        # Get FPS from video_fps_0.json (default to 25 if not found)
        # Try without .mp4 extension first, then with .mp4 extension
        fps = video_fps.get(folder_name, video_fps.get(f"{folder_name}.mp4", 25))
        
        return {"video_name": video_name, "fps": fps}
    except Exception as e:
        print(f"Error getting FPS for video {video_name}: {e}")
        return {"video_name": video_name, "fps": 25, "error": str(e)}

@app.get("/api/surrounding_frames")
async def get_surrounding_frames(
    video_name: str,
    frame_id: str,
    range_val: int = 100
):
    """Get surrounding frames for a given frame"""
    try:
        # Convert frame_id to int (handle both "00026" and "26" formats)
        frame_id_int = int(frame_id)
        
        # Remove .mp4 extension if present
        folder_name = video_name.replace('.mp4', '')
        
        # Extract batch info (L21, L22, L23, etc.)
        batch_part = folder_name[:3]  # e.g., "L21"
        
        # Construct the correct folder structure: Keyframes_L21/keyframes/L21_V001
        keyframes_folder = f"Keyframes_{batch_part}"
        full_folder_path = f"{keyframes_folder}/keyframes/{folder_name}"
        
        # Đường dẫn đến thư mục chứa các frame đã cắt sẵn
        target_dir = os.path.join(KEYFRAME_FOLDER_PATH, full_folder_path)
        print(f"Looking for frames in: {target_dir}")
        
        # Kiểm tra thư mục có tồn tại không
        if not os.path.exists(target_dir):
            print(f"Directory not found: {target_dir}")
            return {"frames": [], "error": f"Directory not found: {target_dir}"}
            
        # Lấy danh sách các file webp trong thư mục
        frame_files = [f for f in os.listdir(target_dir) if f.endswith('.webp')]
        print(f"Found {len(frame_files)} webp files in directory")
        
        # Chuyển tên file thành số frame
        frame_numbers = []
        for frame_file in frame_files:
            try:
                frame_num = int(frame_file.replace('.webp', ''))
                frame_numbers.append(frame_num)
            except ValueError:
                continue
                
        # Sắp xếp các số frame
        frame_numbers.sort()
        print(f"Extracted {len(frame_numbers)} frame numbers: {frame_numbers[:10]}...")
        
        # Tìm các frame xung quanh frame_id
        print(f"Looking for frame_id: {frame_id_int} (type: {type(frame_id_int)})")
        print(f"Available frame numbers: {sorted(frame_numbers)[:10]}... (total: {len(frame_numbers)})")
        
        current_frame_index = -1
        for i, num in enumerate(frame_numbers):
            if num == frame_id_int:
                current_frame_index = i
                print(f"Found exact match for frame_id {frame_id_int} at index {i}")
                break
                
        if current_frame_index == -1:
            # Nếu không tìm thấy frame hiện tại, tìm frame gần nhất
            print(f"Frame {frame_id_int} not found in directory, finding closest frame")
            closest_frame = min(frame_numbers, key=lambda x: abs(x - frame_id_int))
            for i, num in enumerate(frame_numbers):
                if num == closest_frame:
                    current_frame_index = i
                    break
            print(f"Using closest frame {closest_frame} at index {current_frame_index}")
        
        # Lấy các frame xung quanh
        # range_val: số frame trước và sau frame hiện tại trong thư mục keyframes
        start_index = max(0, current_frame_index - range_val)  # range_val frame trước
        end_index = min(len(frame_numbers) - 1, current_frame_index + range_val)  # range_val frame sau
        print(f"Using frame range from index {start_index} to {end_index}")
        print(f"Will show {current_frame_index - start_index} frames before and {end_index - current_frame_index} frames after current frame")
        
        # Tạo kết quả
        frames = []
        for i in range(start_index, end_index + 1):
            frame_num = frame_numbers[i]
            
            # Tạo đường dẫn đầy đủ đến file hình ảnh với leading zeros
            frame_filename = f"{frame_num:05d}.webp"  # Format with 5 digits and leading zeros
            frame_path = os.path.join(target_dir, frame_filename)
            
            # Kiểm tra file có tồn tại không
            if os.path.exists(frame_path):
                # Tạo URL tương đối cho frontend
                # Không thêm tiền tố /siu_sayan_9 vì sẽ được thêm ở phía client
                url = f"/img/{full_folder_path}/{frame_filename}"
                
                frame_info = {
                    "frame_id": frame_num,
                    "is_current": frame_num == frame_id_int,
                    "exists": True,
                    "url": url,
                    "full_path": frame_path  # Thêm thông tin để debug
                }
                print(f"Added frame {frame_num} with URL {url}")
            else:
                frame_info = {
                    "frame_id": frame_num,
                    "is_current": frame_num == frame_id_int,
                    "exists": False
                }
                print(f"Frame file not found: {frame_path}")
                
            frames.append(frame_info)
        
        return {"frames": frames}
    except Exception as e:
        print(f"Error getting surrounding frames: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting surrounding frames: {str(e)}")

@app.post("/submit_dres")
async def submit_dres(request: Request):
    data = await request.json()
    session_id = data.get("session_id")
    evaluation_id = data.get("evaluation_id")
    mediaitem = data.get("mediaitem")
    frame = data.get("frame")
    qa = data.get("qa")
    submission_type = data.get("submission_type", "KIS")
    trake_frame_list = data.get("trake_frame_list", "")

    print(f"🎯 DRES SUBMIT DEBUG:")
    print(f"   📝 mediaitem: '{mediaitem}'")
    print(f"   🖼️ frame: {frame}")
    print(f"   🎯 evaluation_id: {evaluation_id}")
    print(f"   📋 submission_type: {submission_type}")
    
    # Check if mediaitem is from L13-L24 batch
    if mediaitem and len(mediaitem) >= 3:
        batch_num = int(mediaitem[1:3])  # Extract L01 -> 1, L13 -> 13, etc.
        if batch_num >= 13:
            print(f"   ⚠️ BATCH WARNING: Video {mediaitem} is from batch {batch_num} (L13-L24)")
        else:
            print(f"   ✅ BATCH OK: Video {mediaitem} is from batch {batch_num} (L01-L12)")

    # Lấy fps cho video (for KIS and QA only)
    if submission_type in ["KIS", "QA"]:
        fps = video_fps.get(mediaitem)
        if not fps:
            print(f"   ❌ FPS ERROR: No FPS found for '{mediaitem}'")
            print(f"   📋 Available videos in FPS dict: {len(video_fps)} videos")
            # Try with .mp4 extension
            fps_with_ext = video_fps.get(f"{mediaitem}.mp4")
            if fps_with_ext:
                print(f"   ✅ FPS FOUND with .mp4 extension: {fps_with_ext}")
                fps = fps_with_ext
            else:
                print(f"   ❌ FPS ERROR: Not found even with .mp4 extension")
                return JSONResponse({"status": "error", "msg": f"FPS not found for {mediaitem}"})
        else:
            print(f"   ✅ FPS FOUND: {fps}")
        
        try:
            ms = int(int(frame) / float(fps) * 1000)
            print(f"   ⏱️ TIME CALCULATION: frame {frame} / fps {fps} = {ms}ms")
        except Exception as e:
            print(f"   ❌ TIME ERROR: {e}")
            return JSONResponse({"status": "error", "msg": f"Conversion error: {e}"})

    # Đăng nhập nếu chưa có session_id
    if not session_id:
        print("   🔐 LOGIN: Getting new session...")
        session_id = get_session_id()
        if not session_id:
            print("   ❌ LOGIN FAILED")
            return JSONResponse({"status": "error", "msg": "Login failed"})
        else:
            print(f"   ✅ LOGIN SUCCESS: {session_id}")

    # Tạo payload dựa vào submission type
    if submission_type == "KIS":
        payload = {
            "answerSets": [
                {
                    "answers": [
                        {
                            "mediaItemName": mediaitem,
                            "start": ms,
                            "end": ms
                        }
                    ]
                }
            ]
        }
    elif submission_type == "QA":
        # Format: QA-<ANSWER>-<VIDEO_ID>-<TIME(ms)>
        text = f"QA-{qa}-{mediaitem}-{ms}"
        payload = {
            "answerSets": [
                {
                    "answers": [
                        {
                            "text": text
                        }
                    ]
                }
            ]
        }
        print(f"   📝 QA TEXT: {text}")
    elif submission_type == "TRAKE":
        # Format: TR-<VIDEO_ID>-<FRAME_ID1>,<FRAME_ID2>,...
        text = f"TR-{mediaitem}-{trake_frame_list}"
        payload = {
            "answerSets": [
                {
                    "answers": [
                        {
                            "text": text
                        }
                    ]
                }
            ]
        }
        print(f"   📝 TRAKE TEXT: {text}")
    
    print(f"   📤 PAYLOAD: {payload}")
    
    submit_url = f"{DRES_BASE_URL}/api/v2/submit/{evaluation_id}"
    params = {"session": session_id}
    
    print(f"   🌐 SUBMIT URL: {submit_url}")
    print(f"   📋 PARAMS: {params}")
    
    try:
        resp = requests.post(submit_url, json=payload, params=params)
        print(f"   📥 DRES RESPONSE STATUS: {resp.status_code}")
        print(f"   📥 DRES RESPONSE BODY: {resp.text}")
        
        if resp.status_code == 200:
            result = resp.json()
            print(f"   📋 PARSED RESPONSE: {result}")
            
            # Xác định đúng/sai dựa vào trường 'submission'
            submission_status = result.get('submission', '').upper()
            print(f"   🎯 SUBMISSION STATUS: '{submission_status}'")
            
            if submission_status == 'CORRECT':
                print("   ✅ RESULT: CORRECT")
                return JSONResponse({'status': 'success'})
            elif submission_status == 'WRONG':
                print("   ❌ RESULT: WRONG")
                return JSONResponse({'status': 'fail'})
            else:
                print(f"   ⚠️ RESULT: UNKNOWN STATUS '{submission_status}'")
                return JSONResponse({'status': 'submitted', 'raw': result})
        else:
            print(f"   ❌ HTTP ERROR: {resp.status_code}")
            return JSONResponse({"status": "error", "msg": f"Submit error: {resp.status_code} {resp.text}"})
    except Exception as e:
        print(f"   ❌ EXCEPTION: {e}")
        return JSONResponse({"status": "error", "msg": f"Exception: {e}"})

@app.post("/api/cut_frames")
async def cut_frames(request: Request):
    """Cut frames from video and save them"""
    try:
        data = await request.json()
        video_name = data.get("video_name")
        frames = data.get("frames", [])
        
        print(f"Received cut request for video: {video_name}, frames: {frames}")
        
        if not video_name or not frames:
            raise HTTPException(status_code=400, detail="Missing video_name or frames")
        
        # Remove .mp4 extension if present
        folder_name = video_name.replace('.mp4', '')
        
        # Extract batch info (L21, L22, L23, etc.)
        batch_part = folder_name[:3]  # e.g., "L21"
        
        # Construct the correct folder structure: Videos_L21/video/L21_V001.mp4
        video_folder = f"Videos_{batch_part}"
        
        # Đường dẫn video nguồn
        video_path = os.path.join(VIDEOS_FOLDER_PATH, video_folder, "video", f"{video_name}.mp4")
        
        print(f"Video path: {video_path}")
        
        if not os.path.exists(video_path):
            print(f"Video not found at: {video_path}")
            # Thử đường dẫn thay thế
            alt_video_path = os.path.join(VIDEOS_FOLDER_PATH, f"{video_name}.mp4")
            print(f"Trying alternative path: {alt_video_path}")
            
            if os.path.exists(alt_video_path):
                video_path = alt_video_path
            else:
                raise HTTPException(status_code=404, detail=f"Video not found: {video_name}")
        
        # Tạo thư mục lưu trữ nếu chưa tồn tại
        output_dir = os.path.join(script_dir, "static", "cut_frames", video_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Output directory: {output_dir}")
        
        # Lấy FPS của video từ video_fps_0.json
        # Try without .mp4 extension first, then with .mp4 extension
        fps = video_fps.get(folder_name, video_fps.get(f"{folder_name}.mp4", 25))
        print(f"Using FPS: {fps}")
        
        # Cắt từng frame
        results = []
        for frame_id in frames:
            try:
                # Tính thời gian của frame
                timestamp = int(frame_id) / fps
                
                # Tên file đầu ra
                output_file = os.path.join(output_dir, f"{frame_id}.webp")
                
                print(f"Extracting frame {frame_id} at timestamp {timestamp}s to {output_file}")
                
                # Tạo lệnh ffmpeg để cắt frame và chuyển sang webp
                cmd = [
                    "ffmpeg",
                    "-ss", str(timestamp),
                    "-i", video_path,
                    "-vframes", "1",
                    "-q:v", "80",  # Quality for webp (0-100, 80 is good quality)
                    "-f", "webp",   # Output format webp
                    "-y",
                    output_file
                ]
                
                print(f"Running command: {' '.join(cmd)}")
                
                # Thực hiện lệnh
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    print(f"Error extracting frame {frame_id}: {stderr.decode()}")
                    results.append({
                        "frame_id": frame_id,
                        "success": False,
                        "error": "FFmpeg error"
                    })
                else:
                    # Kiểm tra file đã được tạo chưa
                    if os.path.exists(output_file):
                        print(f"Successfully extracted frame {frame_id}")
                        results.append({
                            "frame_id": frame_id,
                            "success": True,
                            "url": f"/siu_sayan_9/static/cut_frames/{video_name}/{frame_id}.webp"
                        })
                    else:
                        print(f"File not created for frame {frame_id}")
                        results.append({
                            "frame_id": frame_id,
                            "success": False,
                            "error": "File not created"
                        })
            except Exception as e:
                print(f"Error processing frame {frame_id}: {e}")
                results.append({
                    "frame_id": frame_id,
                    "success": False,
                    "error": str(e)
                })
        
        # Tạo file zip nếu có ít nhất một frame được cắt thành công
        successful_frames = [r for r in results if r.get("success")]
        zip_url = None
        
        if successful_frames:
            try:
                # Tạo tên file zip
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                zip_filename = f"{video_name}_frames_{timestamp}.zip"
                zip_path = os.path.join(output_dir, zip_filename)
                
                print(f"Creating ZIP file: {zip_path}")
                
                # Tạo file zip
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for result in successful_frames:
                        frame_id = result.get("frame_id")
                        frame_path = os.path.join(output_dir, f"{frame_id}.webp")
                        if os.path.exists(frame_path):
                            zipf.write(frame_path, f"{frame_id}.webp")
                
                zip_url = f"/siu_sayan_9/static/cut_frames/{video_name}/{zip_filename}"
                print(f"ZIP file created: {zip_url}")
            except Exception as e:
                print(f"Error creating ZIP file: {e}")
        
        response_data = {
            "status": "success",
            "video_name": video_name,
            "total_frames": len(frames),
            "successful_frames": len(successful_frames),
            "results": results,
            "zip_url": zip_url
        }
        
        print(f"Returning response: {response_data}")
        return response_data
    except Exception as e:
        print(f"Error cutting frames: {e}")
        raise HTTPException(status_code=500, detail=f"Error cutting frames: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting AIC 2025 SIU Sayan Web Interface...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8609,
        log_level="info"
    )
