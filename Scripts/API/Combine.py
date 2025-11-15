from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
import os
import sys
import time
import logging
import asyncio
import aiohttp
from collections import defaultdict, Counter
import math
import numpy as np
import bisect

# Environment setup
os.environ['HF_HOME'] = '/workspace/competitions/AIC_2025/SIU_Sayan/Base/Cache'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sys.path.append("/workspace/competitions/AIC_2025/SIU_Sayan/Base/Scripts/")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Model Fusion API - FAISS GPU",
    description="API for combining results from multiple FAISS GPU models using temporal fusion and RRF",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API endpoints for different FAISS GPU models
MODEL_ENDPOINTS = {
    "siglip": "http://localhost:8503/text_search",
    "laion": "http://localhost:8502/text_search", 
    "dfn5b": "http://localhost:8501/text_search",
    "llm2clip": "http://localhost:8512/text_search",
    "metaclip": "http://localhost:8511/text_search",
    "metaclip2": "http://localhost:8510/text_search",
    "siglip2": "http://localhost:8513/text_search",
    "jinaclipv2": "http://localhost:8505/text_search"
}

# Pydantic models
class SearchResult(BaseModel):
    keyframe_id: str
    score: float
    video_name: str
    idx_folder: Optional[str] = ""
    key: Optional[str] = ""
    fps: Optional[str] = ""

class SearchResults:
    """Class to store search results for internal processing"""
    def __init__(self, keyframe_id, score, video_name):
        self.keyframe_id = int(keyframe_id)
        self.score = float(score)
        self.video_name = video_name

    def get_img_id(self):
        return self.keyframe_id

    def to_dict(self):
        return {
            'keyframe_id': self.keyframe_id,
            'score': self.score,
            'video_name': self.video_name
        }

    def __repr__(self):
        return f"SearchResults(keyframe_id={self.keyframe_id}, score={self.score}, video_name={self.video_name})"

class TemporalSearchRequest(BaseModel):
    text: str
    model: str = "siglip"
    k: int = 200

class RRFSearchRequest(BaseModel):
    text: str
    k: int = 200
    models: List[str] = ["siglip", "laion", "dfn5b", "llm2clip", "metaclip", "metaclip2", "siglip2", "jinaclipv2"]

class CombineResponse(BaseModel):
    results: List[List[str]]
    total_results: int
    processing_time: float
    method: str

def get_video_hash_map(results, time_quantizer):
    """Creates a hash map of video IDs with quantized times."""
    shmap = defaultdict(dict)
    for result in results:
        # Quantize keyframe_id as a proxy for time
        quantized_time = result.keyframe_id // time_quantizer
        shmap[result.video_name][quantized_time] = result
    return shmap

def combine_results_temporal(results_list, top_k, time_quantizer, time_interval, min_num_lists=1):
    """Combine results using temporal fusion"""
    n_lists_to_merge = len(results_list)
    qi = round(time_interval / time_quantizer)  # Number of intervals to search within

    # Initialize storage for merged results
    merged_results = []

    # Compute max scores for each list, avoid division by zero
    max_scores = [max((result.score for result in results), default=1) for results in results_list]

    # Group results by video
    shmap_list = [get_video_hash_map(results, time_quantizer) for results in results_list]

    # Count occurrences of video IDs
    video_id_counts = Counter()
    for shmap in shmap_list:
        video_id_counts.update(shmap.keys())

    # Include videos that appear in at least 'min_num_lists' result sets
    video_ids = {video_id for video_id, count in video_id_counts.items() if count >= min_num_lists}

    for video_name in video_ids:
        hm_list = [shmap.get(video_name, {}) for shmap in shmap_list]

        # Collect all quantized times for this video
        quantized_times = set()
        for hm in hm_list:
            quantized_times.update(hm.keys())

        for quantized_time in quantized_times:
            best_results = []
            for i in range(n_lists_to_merge):
                matching_result = None
                for offset in range(-qi, qi + 1):
                    matching_result = hm_list[i].get(quantized_time + offset)
                    if matching_result:
                        break
                best_results.append(matching_result)

            # Combine scores
            scores = [result.score / max_scores[i] for i, result in enumerate(best_results) if result]
            if scores:
                combined_score = math.prod(scores)
                if combined_score > 0:
                    keyframe_id = next((result.keyframe_id for result in best_results if result), None)
                    merged_results.append(SearchResults(
                        keyframe_id=keyframe_id,
                        score=combined_score,
                        video_name=video_name
                    ))

    # Sort and return top K results
    merged_results.sort(key=lambda x: x.score, reverse=True)
    return merged_results[:top_k]

def convert_to_search_results(json_response):
    """Helper function to convert JSON response to SearchResults objects"""
    return [SearchResults(d['keyframe_id'], d['score'], d['video_name']) for d in json_response]

def merge_scores(list_res_A, list_res_B):
    """Merge scores for temporal search - forward direction"""
    idx_results = {}
    # Iterate over list_res_B
    for idx_B, record_B in enumerate(list_res_B):
        max_temp_score = 0.0
        video_name = record_B['video_name']
        keyframe_id = record_B['keyframe_id']
        # Iterate over list_res_A to find matching records
        for idx_A, record_A in enumerate(list_res_A):
            # Check if video_name matches and keyframe_id difference is less than 1000
            if (record_A[-1]['video_name'] == record_B['video_name'] and 
                int(record_B['keyframe_id']) - int(record_A[-1]['keyframe_id']) >= 1 and
                int(record_B['keyframe_id']) - int(record_A[-1]['keyframe_id']) <= 1000):
                if float(record_A[-1]['score'])>max_temp_score:
                    max_temp_score = float(record_A[-1]['score'])
                    idx_results[(video_name, keyframe_id)] = idx_A
        
        # Update the score in B
        list_res_B[idx_B]['score'] = float(list_res_B[idx_B]['score']) + max_temp_score
    
    #resort the score
    sorted_list = sorted(list_res_B, key=lambda x: x['score'], reverse=True)
    results = []
    for record_B in sorted_list:
        video_name = record_B['video_name']
        keyframe_id = record_B['keyframe_id']
        if (video_name, keyframe_id) not in idx_results:
            continue
        idx_A = idx_results[(video_name, keyframe_id)]
        record_A = list_res_A[idx_A]
        results.append(record_A + [record_B])

    max_dict = {}
    for item in results:
        key = str(item[-2]["video_name"]) + "_" + str(item[-2]["keyframe_id"])
        if key not in max_dict or item[-1]["score"] > max_dict[key][-1]["score"]:
            max_dict[key] = item
    results = list(max_dict.values())
    
    return results

def merge_scores_reverse(list_res_A, list_res_B):
    """Merge scores for temporal search - reverse direction"""
    idx_results = {}
    # Iterate over list_res_A
    for idx_A, record_A in enumerate(list_res_A):
        max_temp_score = 0.0
        video_name = record_A['video_name']
        keyframe_id = record_A['keyframe_id']
        # Iterate over list_res_A to find matching records
        for idx_B, record_B in enumerate(list_res_B):
            # Check if video_name matches and keyframe_id difference is less than 1000
            if (record_A['video_name'] == record_B[0]['video_name'] and 
                int(record_B[0]['keyframe_id']) - int(record_A['keyframe_id']) >= 1 and
                int(record_B[0]['keyframe_id']) - int(record_A['keyframe_id']) <= 1000):
                if float(record_B[0]['score'])>max_temp_score:
                    max_temp_score = float(record_B[0]['score'])
                    idx_results[(video_name, keyframe_id)] = idx_B
        
        # Update the score in B
        list_res_A[idx_A]['score'] = float(record_A['score']) + max_temp_score
    
    #resort the score
    sorted_list = sorted(list_res_A, key=lambda x: x['score'], reverse=True)
    results = []
    for record_A in sorted_list:
        video_name = record_A['video_name']
        keyframe_id = record_A['keyframe_id']
        if (video_name, keyframe_id) not in idx_results:
            continue
        idx_B = idx_results[(video_name, keyframe_id)]
        record_B = list_res_B[idx_B]
        results.append([record_A] + record_B)

    max_dict = {}
    for item in results:
        key = str(item[1]["video_name"]) + "_" + str(item[1]["keyframe_id"])
        if key not in max_dict or item[0]["score"] > max_dict[key][0]["score"]:
            max_dict[key] = item
    results = list(max_dict.values())
    
    return results

def combine_results_temporal_advanced(results_list, top_k, time_quantizer=200.0, time_interval=300.0, min_num_lists=1):
    """
    Advanced temporal search using merge_scores functions from Temporal directory
    This implements the same logic as the Temporal/Vector_database/faiss_gpu.py
    """
    if not results_list or len(results_list) < 2:
        # If only one result set, return it directly
        if results_list:
            return [[result] for result in results_list[0][:top_k]]
        return []
    
    # Convert results to the format expected by merge_scores functions
    formatted_results = []
    for result_set in results_list:
        formatted_set = []
        for result in result_set:
            formatted_result = {
                'video_name': result['video_name'],
                'keyframe_id': result['keyframe_id'],
                'score': result['score']
            }
            formatted_set.append(formatted_result)
        formatted_results.append(formatted_set)
    
    # Apply temporal merging using the advanced merge_scores functions
    merged_results = []
    
    # Forward direction merging
    if len(formatted_results) >= 2:
        current_merged = formatted_results[0]
        for i in range(1, len(formatted_results)):
            current_merged = merge_scores(current_merged, formatted_results[i])
        merged_results.extend(current_merged)
    
    # Reverse direction merging for better coverage
    if len(formatted_results) >= 2:
        current_merged_reverse = formatted_results[-1]
        for i in range(len(formatted_results) - 2, -1, -1):
            current_merged_reverse = merge_scores_reverse(formatted_results[i], current_merged_reverse)
        merged_results.extend(current_merged_reverse)
    
    # Remove duplicates and sort by score
    unique_results = {}
    for result in merged_results:
        if isinstance(result, list) and len(result) > 0:
            # Handle list format from merge_scores
            key = f"{result[-1]['video_name']}_{result[-1]['keyframe_id']}"
            if key not in unique_results or result[-1]['score'] > unique_results[key][-1]['score']:
                unique_results[key] = result
        else:
            # Handle single result format
            key = f"{result['video_name']}_{result['keyframe_id']}"
            if key not in unique_results or result['score'] > unique_results[key]['score']:
                unique_results[key] = [result]
    
    # Sort by score and return top_k
    sorted_results = sorted(unique_results.values(), 
                          key=lambda x: float(x[-1]['score']) if isinstance(x, list) else float(x['score']), 
                          reverse=True)
    
    return sorted_results[:top_k]

def merge_results_rrf_with_boost(results_list, top_k, k_rrf=100, n=3):
    """
    Merges multiple search result sets using Reciprocal Rank Fusion (RRF) with score boosting
    """
    fused_scores = defaultdict(lambda: {'score': 0.0, 'video_name': None})
    n_hits = len(results_list)  # Number of result sets

    def process_results(result_set):
        max_rank = min(top_k, len(result_set))
        for rank, result in enumerate(result_set[:max_rank]):
            keyframe_id = result['keyframe_id']
            score = float(result['score'])
            video_name = result['video_name']
            # Compute RRF score
            rrf_score = k_rrf / (rank + k_rrf)
            # Boost the score of the top-n results
            if rank < n:
                rrf_score *= n_hits
            # Accumulate score and store video_name
            fused_scores[keyframe_id]['score'] += rrf_score
            if fused_scores[keyframe_id]['video_name'] is None:
                fused_scores[keyframe_id]['video_name'] = video_name

    # Process all result sets
    for result_set in results_list:
        process_results(result_set)

    # Sort by score (descending) and return the top_k results
    sorted_results = sorted(fused_scores.items(), key=lambda x: -x[1]['score'])
    return [(keyframe_id, data['video_name'], data['score']) for keyframe_id, data in sorted_results[:top_k]]

async def query_model_api(session, model_name, text, k):
    """Query a specific model API - simplified without batch filtering"""
    try:
        if model_name not in MODEL_ENDPOINTS:
            logger.error(f"Unknown model: {model_name}")
            return None
            
        url = MODEL_ENDPOINTS[model_name]
        params = {
            "text": text,
            "k": k
        }
        
        logger.info(f"Querying {model_name} API: {url}")
        async with session.get(url, params=params, timeout=30) as response:
            if response.status == 200:
                data = await response.json()
                logger.info(f"✅ {model_name} returned {len(data)} results")
                return data
            else:
                logger.error(f"❌ {model_name} API error: {response.status}")
                return None
    except Exception as e:
        logger.error(f"❌ Error querying {model_name}: {str(e)}")
        return None

@app.post("/search_temporal", response_model=CombineResponse)
async def search_temporal(request: TemporalSearchRequest):
    """Search using temporal fusion"""
    start_time = time.time()
    
    try:
        # Split text into parts for temporal queries
        text_parts = [i.strip() for i in request.text.split('.') if i.strip()]
        if not text_parts:
            raise HTTPException(status_code=400, detail="No valid text parts provided")

        temporal_queries = []
        
        async with aiohttp.ClientSession() as session:
            for text_part in text_parts:
                result = await query_model_api(
                    session, request.model, text_part, request.k
                )
                if result:
                    temporal_queries.append(convert_to_search_results(result))
                else:
                    logger.warning(f"No results for text part: '{text_part}'")

        if not temporal_queries:
            raise HTTPException(status_code=404, detail="No results from queries")

        # Combine results using temporal fusion (same as working old code)
        temporal_ans = combine_results_temporal(
            temporal_queries,
            top_k=request.k,
            time_quantizer=200.0,
            time_interval=300.0,
            min_num_lists=1
        )
        
        results = [[res.video_name, str(res.keyframe_id)] for res in temporal_ans]
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Temporal search completed: {len(results)} results in {processing_time:.2f}s")
        
        return CombineResponse(
            results=results,
            total_results=len(results),
            processing_time=processing_time,
            method="temporal_fusion"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in temporal search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/search_rrf", response_model=CombineResponse)
async def search_rrf(request: RRFSearchRequest):
    """Search using Reciprocal Rank Fusion"""
    start_time = time.time()
    
    try:
        results = []
        
        async with aiohttp.ClientSession() as session:
            # Query all models in parallel
            tasks = []
            for model_name in request.models:
                if model_name in MODEL_ENDPOINTS:
                    task = query_model_api(
                        session, model_name, request.text, request.k
                    )
                    tasks.append(task)
            
            # Wait for all queries to complete
            model_results = await asyncio.gather(*tasks)
            
            # Filter out None results
            results = [result for result in model_results if result is not None]

        if not results:
            raise HTTPException(status_code=404, detail="No results from any model")

        # Merge results using RRF
        merged_results = merge_results_rrf_with_boost(results, request.k, 100, 3)
        
        results_formatted = [[res[1], res[0]] for res in merged_results]
        processing_time = time.time() - start_time
        
        logger.info(f"✅ RRF search completed: {len(results_formatted)} results in {processing_time:.2f}s")
        
        return CombineResponse(
            results=results_formatted,
            total_results=len(results_formatted),
            processing_time=processing_time,
            method="reciprocal_rank_fusion"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in RRF search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/search_temporal")
async def search_temporal_get(
    text: str = Query(..., description="Text query"),
    model: str = Query("siglip", description="Model to use"),
    k: int = Query(200, description="Number of results")
):
    """GET endpoint for temporal search"""
    try:
        request = TemporalSearchRequest(
            text=text, model=model, k=k
        )
        return await search_temporal(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameters: {str(e)}")

@app.get("/search_rrf")
async def search_rrf_get(
    text: str = Query(..., description="Text query"),
    k: int = Query(200, description="Number of results"),
    models: str = Query("['siglip','laion','dfn5b','llm2clip','metaclip','metaclip2','siglip2','jinaclipv2']", description="Models to use")
):
    """GET endpoint for RRF search"""
    try:
        models_list = eval(models) if models else ["siglip", "laion", "dfn5b", "llm2clip", "metaclip", "metaclip2", "siglip2", "jinaclipv2"]
        request = RRFSearchRequest(
            text=text, k=k, models=models_list
        )
        return await search_rrf(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameters: {str(e)}")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_endpoints": MODEL_ENDPOINTS,
        "available_methods": ["temporal_fusion", "reciprocal_rank_fusion"]
    }

@app.get("/")
async def index():
    """Root endpoint with API information"""
    return {
        "message": "Model Fusion API",
        "version": "1.0.0",
        "endpoints": {
            "POST /search_temporal": "Temporal fusion search",
            "POST /search_rrf": "Reciprocal Rank Fusion search",
            "GET /search_temporal": "Temporal fusion search (GET)",
            "GET /search_rrf": "RRF search (GET)",
            "GET /health": "Health check",
            "GET /": "This endpoint"
        },
        "supported_models": list(MODEL_ENDPOINTS.keys()),
        "fusion_methods": ["temporal_fusion", "reciprocal_rank_fusion"]
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Model Fusion API server...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8504,
        log_level="info"
    )
