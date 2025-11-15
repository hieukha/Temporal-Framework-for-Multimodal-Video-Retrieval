from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import time
import os
import numpy as np
import ujson
import subprocess
import sys

# Environment setup
os.environ['HF_HOME'] = '/workspace/competitions/AIC_2025/SIU_Sayan/Base/Cache'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Add the Scripts directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Extract_Feature.Class.DFN5B import DFN5B
from Vector_database.faiss_gpu import FAISS_GPU

# Initialize feature extraction model
print("🔄 Loading DFN5B model...")
start_time = time.time()
model = DFN5B()
model_load_time = time.time() - start_time
print(f"✅ DFN5B model loaded in {model_load_time:.2f} seconds")

# Configuration
FEATURES_PATH = ["/dataset/AIC_2025/SIU_Sayan/autoshot/features_dfn5b"]
SPLIT_NAME = ["autoshot"]

# Initialize FAISS GPU connection with HIGH_ACCURACY for best results
print("🔄 Initializing FAISS GPU connection...")
start_time = time.time()
faiss_db = FAISS_GPU("dfn5b_v3", index_type="HIGH_ACCURACY")

# Create/use the dfn5b_v3 collection
print("🔗 Setting up 'dfn5b_v3' collection...")
# Check if index exists, if not create it
if not faiss_db.is_trained:
    print("FAISS index does not exist. Creating...")
    faiss_db.addDatabase("dfn5b_v3", 1024, FEATURES_PATH, SPLIT_NAME)
else:
    print("FAISS index already exists. Using existing index.")

faiss_init_time = time.time() - start_time
print(f"✅ FAISS GPU initialized in {faiss_init_time:.2f} seconds")

# Dummy query to warm up the system
print("🔄 Warming up system...")
start_time = time.time()

if faiss_db.is_trained:
    dummy_query_path = "/workspace/competitions/AIC_2025/SIU_Sayan/Base/Cache/dummy_query_dfn5b.npy"
    if os.path.exists(dummy_query_path):
        dummy_query = np.load(dummy_query_path).reshape(1,-1).astype('float32')[0]
        faiss_db.search(dummy_query, 3)
        print("Dummy Query Finished")
    else:
        print("Creating dummy query...")
        dummy_query = np.random.random(1024).astype('float32')
        np.save(dummy_query_path, dummy_query)
        print("Dummy query created")
else:
    print("Index not trained, skipping warmup")

warmup_time = time.time() - start_time
print(f"✅ System warmed up in {warmup_time:.2f} seconds")

def preprocessing_text(text: str) -> np.ndarray:
    """Preprocess text to feature vector using DFN5B model"""
    global model
    text_feat_arr = model.get_text_features(text) 
    
    # Ensure it's a numpy array
    if hasattr(text_feat_arr, 'cpu'):
        text_feat_arr = text_feat_arr.cpu().detach().numpy()
    
    # Convert to numpy array if it's not already
    if not isinstance(text_feat_arr, np.ndarray):
        text_feat_arr = np.array(text_feat_arr)
    
    text_feat_arr = text_feat_arr.reshape(1,-1).astype(np.float32)
    return text_feat_arr[0]

def preprocessing_image(image: str) -> np.ndarray:
    """Preprocess image to feature vector using DFN5B model"""
    global model
    image_feat_arr = model.get_image_features(image) 
    
    # Ensure it's a numpy array
    if hasattr(image_feat_arr, 'cpu'):
        image_feat_arr = image_feat_arr.cpu().detach().numpy()
    
    # Convert to numpy array if it's not already
    if not isinstance(image_feat_arr, np.ndarray):
        image_feat_arr = np.array(image_feat_arr)
    
    image_feat_arr = image_feat_arr.reshape(1,-1).astype(np.float32)
    return image_feat_arr[0]

# Shot dictionary cache
list_shot_dict = {}

def getShot(videoName: str, frameName: int):
    """Get shot information for a specific video and frame"""
    if videoName not in list_shot_dict.keys():
        # New structure: files are directly in SceneJson folder, not in subfolders
        json_path = os.path.join("/dataset/AIC_2025/SIU_Sayan/autoshot/SceneJson", f"{videoName}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as file:
                data = ujson.load(file)
            list_shot_dict[videoName] = data
        else:
            # Return default shot if file doesn't exist
            return [[["00000.webp", "99999.webp"]]]
    else:
        data = list_shot_dict[videoName]
    
    for i, list_frame in enumerate(data):
        if int(list_frame[0][0][:-4]) > int(frameName):
            return data[i-1]
    return data[len(data)-1]

def sort_by_shot(results: List[dict]) -> List[dict]:
    """Sort search results by shot grouping"""
    shot_groups = {}
    image_index = {}

    # Create a mapping of each frame path to its original index
    for index, result in enumerate(results):
        frame_path = result['video_name']
        image_index[frame_path] = index

    # Group frames by shots using getShot
    for result in results:
        video_name = result['video_name'].replace('.mp4','')
        keyframe_id = result['keyframe_id']
        keyframe_number = int(keyframe_id)

        # Get the shot information using the video name and keyframe number
        shot = getShot(video_name, keyframe_number)

        # Convert the shot list to a tuple to make it hashable
        # Handle nested lists properly
        try:
            if shot and len(shot) > 0 and len(shot[0]) > 0:
                # Flatten the nested structure and convert to tuple
                shot_data = shot[0]
                if isinstance(shot_data[0], list):
                    # If it's nested like [[frame1, frame2]]
                    shot_key = tuple(shot_data[0])
                else:
                    # If it's like [frame1, frame2]
                    shot_key = tuple(shot_data)
            else:
                shot_key = (f"default_{video_name}_{keyframe_id}",)
        except (IndexError, TypeError):
            shot_key = (f"default_{video_name}_{keyframe_id}",)

        # Create a group for each shot if not already present
        if shot_key not in shot_groups:
            shot_groups[shot_key] = []
        shot_groups[shot_key].append(result)

    # Sort the groups by the minimum index of the frames within each group
    sorted_groups = sorted(
        shot_groups.values(),
        key=lambda group: min(image_index[result['video_name']] for result in group)
    )

    # Flatten the sorted groups back into a single list of results
    sorted_results = [result for group in sorted_groups for result in group]
    return sorted_results

# Pydantic models for request/response validation
class TextSearchRequest(BaseModel):
    text: str
    k: int = 200

class ImageSearchRequest(BaseModel):
    image_url: str
    k: int = 200

class PreprocessRequest(BaseModel):
    text: str
    k: Optional[str] = ""

class SearchResult(BaseModel):
    key: str
    idx_folder: str
    video_name: str
    keyframe_id: str
    fps: str
    score: str

class HealthResponse(BaseModel):
    status: str
    model: str
    faiss_collection: str
    total_vectors: int
    features_path: str
    index_type: str

# FastAPI app initialization
app = FastAPI(
    title="AIC 2025 SIU Sayan Video Search API - DFN5B (FAISS GPU)",
    description="Advanced video search API using DFN5B model with FAISS GPU vector database",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/preprocess", response_model=List[float])
async def preprocess(request: PreprocessRequest):
    """Preprocess text to feature vector"""
    start_time = time.time()
    
    text = request.text
    if text and text[-1] == '.': 
        text = text[:-1]
    
    text_feat_arr = preprocessing_text(text)
    
    processing_time = time.time() - start_time
    print(f"⏱️  POST /preprocess: '{text}' processed in {processing_time:.3f}s")
    
    return text_feat_arr.tolist()

@app.get("/preprocess", response_model=List[float])
async def preprocess_get(
    text: str = Query(..., description="Text to preprocess"),
    k: Optional[str] = Query("", description="Optional parameter")
):
    """Preprocess text to feature vector (GET method)"""
    start_time = time.time()
    
    if text and text[-1] == '.': 
        text = text[:-1]
    
    text_feat_arr = preprocessing_text(text)
    
    processing_time = time.time() - start_time
    print(f"⏱️  GET /preprocess: '{text}' processed in {processing_time:.3f}s")
    
    return text_feat_arr.tolist()

@app.post("/text_search", response_model=List[SearchResult])
async def text_search(request: TextSearchRequest):
    """Search videos by text description"""
    start_time = time.time()
    
    global faiss_db
    
    # Preprocess text
    text = request.text
    if text and text[-1] == '.': 
        text = text[:-1]
    
    text_feat_arr = preprocessing_text(text)
    
    # Search using FAISS
    search_results = faiss_db.search(
        text_feat_arr, 
        request.k
    )
    
    search_time = time.time() - start_time
    
    print(f"🔍 POST /text_search: '{text}' (k={request.k}) completed in {search_time:.3f}s - Found {len(search_results)} results")
    
    return search_results

@app.get("/text_search", response_model=List[SearchResult])
async def text_search_get(
    text: str = Query(..., description="Search text query"),
    k: int = Query(200, description="Number of results to return")
):
    """Search videos by text description (GET method)"""
    start_time = time.time()
    
    global faiss_db
    
    # Preprocess text
    if text and text[-1] == '.': 
        text = text[:-1]
    
    text_feat_arr = preprocessing_text(text)
    
    # Search using FAISS
    search_results = faiss_db.search(
        text_feat_arr, 
        k
    )
    
    search_time = time.time() - start_time
    
    print(f"🔍 GET /text_search: '{text}' (k={k}) completed in {search_time:.3f}s - Found {len(search_results)} results")
    
    return search_results

@app.post("/image_search", response_model=List[SearchResult])
async def image_search(request: ImageSearchRequest):
    """Search videos by image similarity"""
    start_time = time.time()
    
    global faiss_db
    
    # Preprocess image
    image_feat_arr = preprocessing_image(request.image_url)
    
    # Search using FAISS
    search_results = faiss_db.search(
        image_feat_arr, 
        request.k
    )
    
    search_time = time.time() - start_time
    
    print(f"🔍 POST /image_search: '{request.image_url}' (k={request.k}) completed in {search_time:.3f}s - Found {len(search_results)} results")
    
    return search_results

@app.get("/image_search", response_model=List[SearchResult])
async def image_search_get(
    image_url: str = Query(..., description="URL of image to search"),
    k: int = Query(200, description="Number of results to return")
):
    """Search videos by image similarity (GET method)"""
    start_time = time.time()
    
    global faiss_db
    
    # Preprocess image
    image_feat_arr = preprocessing_image(image_url)
    
    # Search using FAISS
    search_results = faiss_db.search(
        image_feat_arr, 
        k
    )
    
    search_time = time.time() - start_time
    
    print(f"🔍 GET /image_search: '{image_url}' (k={k}) completed in {search_time:.3f}s - Found {len(search_results)} results")
    
    return search_results

@app.get("/health", response_model=HealthResponse)
async def health():
    """Get API health and statistics"""
    global faiss_db
    
    vector_count = faiss_db.getCount() if faiss_db.is_trained else 0
    
    return HealthResponse(
        status="healthy",
        model="DFN5B",
        faiss_collection=faiss_db.collection_name or "none",
        total_vectors=vector_count,
        features_path=str(FEATURES_PATH),
        index_type=faiss_db.index_type
    )

@app.get("/")
async def index():
    """Root endpoint with API information"""
    return {
        "message": "AIC 2025 SIU Sayan Video Search API - DFN5B (FAISS GPU)",
        "version": "1.0.0",
        "endpoints": {
            "text_search": "/text_search",
            "image_search": "/image_search", 
            "preprocess": "/preprocess",
            "health": "/health",
            "docs": "/docs"
        },
        "description": "Advanced video search using DFN5B model with FAISS GPU acceleration",
        "index_type": faiss_db.index_type,
        "feature_size": 1024
    }

if __name__ == "__main__":
    import uvicorn
    
    print(f"🚀 Starting DFN5B FAISS GPU API server on port 8501...")
    uvicorn.run(app, host="0.0.0.0", port=8501) 