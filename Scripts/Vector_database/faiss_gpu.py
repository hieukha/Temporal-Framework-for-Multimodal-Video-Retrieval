import faiss
import numpy as np
import os
import ujson
import time
import pickle
from datetime import datetime
from typing import List, Dict, Optional

DATASET_PATH_TEAM = "/dataset/AIC_2025/SIU_Sayan/keyframes"
DATASET_INDEX = "/dataset/AIC_2025/SIU_Sayan/autoshot/index"

dict_fps_0 = '/dataset/AIC_2025/SIU_Sayan/autoshot/video_fps_0.json'

with open(dict_fps_0, encoding='utf-8-sig') as json_file:
    dict_fps = ujson.load(json_file)
print("FPS Dict loaded")

def get_split(video_name):
    # Updated for new naming convention: batch 0 = L21-L30, batch 1 = L31+
    level_num = int(video_name[1:3])
    if level_num <= 30:
        return "0"  # Keyframes_L21 -> Keyframes_L30
    else:
        return "1"  # Keyframes_L31 trở đi

def get_all_videos():
    """
    Get all available videos from the features directory
    """
    # Get all video files from features directory
    features_dir = "/dataset/AIC_2025/SIU_Sayan/autoshot/features_siglip2"  # Fixed: added missing "p"
    video_list = []
    if os.path.exists(features_dir):
        for file in os.listdir(features_dir):
            if file.endswith('.npy'):
                video_list.append(file.replace('.npy', ''))
    return video_list

def frame_to_index(video, frame):
    with open(DATASET_INDEX + "/" + video + ".json", 'r') as infile:
        index_dict = ujson.load(infile)
    return index_dict[str(int(frame))]

def get_frame_dict(video):
    with open(DATASET_INDEX + "/" + video + ".json", 'r') as infile:
        index_dict = ujson.load(infile)
    return index_dict

def get_index_dict(video):
    with open(DATASET_INDEX + "/" + video + ".json", 'r') as infile:
        index_dict = ujson.load(infile)
    index_dict = {value: key for key, value in index_dict.items()}
    return index_dict

class FAISS_GPU:
    def __init__(self, collection_name=None, index_type="HIGH_ACCURACY", temp_memory_gb=None):
        """
        Initialize FAISS GPU with different index types for different accuracy/speed tradeoffs
        
        Args:
            collection_name: Name of the collection
            index_type: Type of index to use
                - "HIGH_ACCURACY": IndexFlatIP (100% accuracy, slower)
                - "BALANCED": IndexIVFFlat (high accuracy, good speed)
                - "FAST": IndexIVFPQ (lower accuracy, fastest)
                - "HNSW": IndexHNSWFlat (balanced accuracy/speed)
            temp_memory_gb: GPU temp memory in GB (default: None for dynamic allocation)
        
        Note: 
            - Use CUDA_VISIBLE_DEVICES environment variable to control which GPU to use
            - FAISS will use the first available GPU (GPU 0 in CUDA_VISIBLE_DEVICES context)
            - Example: CUDA_VISIBLE_DEVICES=2 python script.py (uses GPU 2 as GPU 0)
        """
        # Initialize FAISS GPU
        self.device = faiss.StandardGpuResources()
        
        # Configure GPU memory - only set TempMemory if explicitly requested
        if temp_memory_gb is not None and temp_memory_gb > 0:
            temp_memory_bytes = int(temp_memory_gb * 1024 * 1024 * 1024)
            self.device.setTempMemory(temp_memory_bytes)
            print(f"FAISS GPU Connection Success")
            print(f"Configured with {temp_memory_gb}GB temp memory for optimal search speed")
        else:
            print("FAISS GPU Connection Success")
            print("Using dynamic memory allocation (memory efficient)")
        
        self.collection_name = collection_name
        self.index = None
        self.metadata = []  # Store metadata for each vector
        self.is_trained = False
        self.index_type = index_type
        self.size = None
        self.temp_memory_gb = temp_memory_gb
        
        # Create cache directory for saving/loading indices
        self.cache_dir = f"/workspace/competitions/AIC_2025/SIU_Sayan/Base/Cache/faiss_indices"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Try to load existing index if available
        if collection_name:
            self._load_index()

    def _get_index_path(self):
        """Get the file path for saving/loading the index"""
        return os.path.join(self.cache_dir, f"{self.collection_name}_{self.index_type}.index")
    
    def _get_metadata_path(self):
        """Get the file path for saving/loading metadata"""
        return os.path.join(self.cache_dir, f"{self.collection_name}_{self.index_type}_metadata.pkl")

    def _create_index(self, feature_size: int):
        """Create FAISS index based on specified type"""
        print(f"Creating FAISS {self.index_type} index...")
        
        if self.index_type == "HIGH_ACCURACY":
            # Flat index for 100% accuracy (brute force)
            cpu_index = faiss.IndexFlatIP(feature_size)
            print("Created IndexFlatIP (100% accuracy)")
            
        elif self.index_type == "BALANCED":
            # IVF with more clusters for higher accuracy
            nlist = min(8192, max(1024, int(np.sqrt(1000000))))  # Adaptive nlist
            cpu_index = faiss.IndexIVFFlat(
                faiss.IndexFlatIP(feature_size),
                feature_size,
                nlist,
                faiss.METRIC_INNER_PRODUCT
            )
            print(f"Created IndexIVFFlat with {nlist} clusters (high accuracy)")
            
        elif self.index_type == "FAST":
            # IVF with Product Quantization for speed
            nlist = 4096
            m = 64  # Number of subquantizers
            cpu_index = faiss.IndexIVFPQ(
                faiss.IndexFlatIP(feature_size),
                feature_size,
                nlist,
                m,
                8,  # Number of bits per subquantizer
                faiss.METRIC_INNER_PRODUCT
            )
            print(f"Created IndexIVFPQ with {nlist} clusters and {m} subquantizers (fast)")
            
        elif self.index_type == "HNSW":
            # HNSW for balanced accuracy/speed
            M = 32  # Number of connections per element
            cpu_index = faiss.IndexHNSWFlat(feature_size, M, faiss.METRIC_INNER_PRODUCT)
            cpu_index.hnsw.efConstruction = 200  # Construction time parameter
            cpu_index.hnsw.efSearch = 64  # Search time parameter
            print(f"Created IndexHNSWFlat with M={M} (balanced)")
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Move to GPU (except HNSW which doesn't support GPU well)
        if self.index_type == "HNSW":
            print("⚠️  HNSW index will run on CPU (better performance than GPU for HNSW)")
            self.index = cpu_index
        else:
            try:
                # Use first available GPU (controlled by CUDA_VISIBLE_DEVICES)
                self.index = faiss.index_cpu_to_gpu(self.device, 0, cpu_index)
                print(f"✅ Index moved to GPU (first available GPU)")
            except Exception as e:
                print(f"⚠️  Failed to move to GPU, using CPU: {e}")
                self.index = cpu_index
        
        return self.index

    def _load_index(self):
        """Load existing FAISS index and metadata if available"""
        index_path = self._get_index_path()
        metadata_path = self._get_metadata_path()
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            print(f"Loading existing FAISS {self.index_type} index: {index_path}")
            try:
                # Load CPU index first
                cpu_index = faiss.read_index(index_path)
                
                # Move to GPU (except HNSW)
                if self.index_type == "HNSW":
                    self.index = cpu_index
                else:
                    try:
                        # Use first available GPU (controlled by CUDA_VISIBLE_DEVICES)
                        self.index = faiss.index_cpu_to_gpu(self.device, 0, cpu_index)
                        print(f"✅ Index loaded to GPU (first available GPU)")
                    except Exception as e:
                        print(f"⚠️  Failed to move to GPU, using CPU: {e}")
                        self.index = cpu_index
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                
                # Get feature size from metadata if available
                if self.metadata:
                    if 'feature_size' in self.metadata[0]:
                        self.size = self.metadata[0]['feature_size']
                
                self.is_trained = True
                print(f"Successfully loaded {self.index_type} index with {len(self.metadata)} vectors")
                return True
            except Exception as e:
                print(f"Error loading index: {e}")
                self.index = None
                self.metadata = []
                self.is_trained = False
                return False
        return False

    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        index_path = self._get_index_path()
        metadata_path = self._get_metadata_path()
        
        try:
            print(f"Saving FAISS {self.index_type} index to disk...")
            
            # Move index to CPU for saving (if it's on GPU)
            if self.index_type != "HNSW" and hasattr(self.index, 'getDevice'):
                cpu_index = faiss.index_gpu_to_cpu(self.index)
            else:
                cpu_index = self.index
                
            faiss.write_index(cpu_index, index_path)
            
            # Save metadata with feature size
            metadata_with_info = []
            for meta in self.metadata:
                meta_copy = meta.copy()
                meta_copy['feature_size'] = self.size
                metadata_with_info.append(meta_copy)
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata_with_info, f)
            
            print(f"Index saved successfully to {index_path}")
        except Exception as e:
            print(f"Error saving index: {e}")

    def addDatabase(self, collection_name, feature_size, FEATURES_PATH, SPLIT_NAME):
        """Create and populate FAISS index"""
        self.collection_name = collection_name
        self.size = feature_size
        
        # Create the appropriate index type
        self._create_index(feature_size)
        
        print("Starting data collection and training...")
        
        # Initialize variables for batch processing
        BATCH_SIZE = 50000  # Larger batch size for FAISS
        all_vectors = []
        all_metadata = []
        total_vectors = 0
        start_time = time.time()
        
        # First pass: collect all vectors and metadata
        print("Collecting vectors and metadata...")
        for idx_folder, folder_path in enumerate(FEATURES_PATH):
            print(f"Processing folder {idx_folder + 1}/{len(FEATURES_PATH)}: {folder_path}")
            
            for feat_npy in sorted(os.listdir(folder_path)):
                video_name = feat_npy.split('.')[0]
                feats_arr = np.load(os.path.join(folder_path, feat_npy))
                video_name_no_ext = video_name.split(".")[0]
                
                # Handle path structure - new structure with Keyframes_L21, Keyframes_L22, etc.
                video_level = video_name_no_ext.split("_")[0]  # L21, L22, etc.
                video_number = video_name_no_ext.split("_")[1]  # V001, V002, etc.
                
                # New structure: /dataset/AIC_2025/SIU_Pumpking/0/frames/autoshot/Keyframes_L21/keyframes/L21_V001/
                frame_path = os.path.join(DATASET_PATH_TEAM, f"Keyframes_{video_level}", "keyframes", video_name_no_ext)
                
                if not os.path.exists(frame_path):
                    print(f"Warning: Frame path not found for {video_name_no_ext} at {frame_path}")
                    continue
                    
                frame_list = sorted(os.listdir(frame_path))
                print(f"  Processing video: {video_name} ({len(feats_arr)} frames)")
                
                for idx, feat in enumerate(feats_arr):
                    # Ensure feat is numpy array and convert data type properly
                    if hasattr(feat, 'cpu'):
                        feat = feat.cpu().detach().numpy()
                    
                    if not isinstance(feat, np.ndarray):
                        feat = np.array(feat)
                    
                    feat_reshaped = feat.reshape(1,-1).astype(np.float32)[0]
                    
                    # Normalize vector for cosine similarity (since we use METRIC_INNER_PRODUCT)
                    feat_normalized = feat_reshaped / np.linalg.norm(feat_reshaped)
                    
                    all_vectors.append(feat_normalized)
                    
                    metadata = {
                        "idx_folder": idx_folder,
                        "video_name": video_name + ".mp4",
                        "frame_name": frame_list[idx].replace(".jpg",""),  # Store original filename without .jpg
                        "fps": dict_fps[video_name]
                    }
                    all_metadata.append(metadata)
                    total_vectors += 1
                    
                    # Process in batches to avoid memory issues
                    if len(all_vectors) >= BATCH_SIZE:
                        print(f"Collected {len(all_vectors)} vectors, continuing...")
        
        print(f"Total vectors collected: {total_vectors:,}")
        
        # Convert to numpy array
        print("Converting to numpy array...")
        vectors_array = np.array(all_vectors, dtype=np.float32)
        self.metadata = all_metadata
        
        # Train the index if needed
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            print(f"Training FAISS {self.index_type} index...")
            if self.index_type == "BALANCED":
                train_size = min(len(vectors_array), max(100000, self.index.nlist * 100))
            elif self.index_type == "FAST":
                train_size = min(len(vectors_array), max(100000, self.index.nlist * 100))
            else:
                train_size = len(vectors_array)
                
            train_vectors = vectors_array[:train_size]
            self.index.train(train_vectors)
            print("Training completed!")
        
        self.is_trained = True
        
        # Add all vectors to index
        print("Adding vectors to index...")
        batch_size = 10000 if self.index_type != "HIGH_ACCURACY" else 5000  # Smaller batches for brute force
        
        for i in range(0, len(vectors_array), batch_size):
            end_idx = min(i + batch_size, len(vectors_array))
            batch_vectors = vectors_array[i:end_idx]
            
            self.index.add(batch_vectors)
            
            progress = (end_idx / len(vectors_array)) * 100
            print(f"Added {end_idx:,}/{len(vectors_array):,} vectors ({progress:.1f}%)")
        
        total_time = time.time() - start_time
        print(f"\n=== FAISS {self.index_type} INDEX CREATION COMPLETED ===")
        print(f"Total vectors indexed: {total_vectors:,}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average speed: {total_vectors/total_time:.0f} vectors/second")
        print(f"Index type: {self.index_type}")
        
        # Save the index to disk
        self._save_index()
        
        return {"status": "success", "total_vectors": total_vectors, "index_type": self.index_type}

    def search(self, query, k, video_list=None):
        """Search using FAISS index"""
        if not self.is_trained or self.index is None:
            raise ValueError("Index is not trained or loaded. Please run addDatabase first.")
        
        # Normalize query vector for cosine similarity
        query_normalized = query / np.linalg.norm(query)
        query_vector = query_normalized.reshape(1, -1).astype(np.float32)
        
        # Get all available videos for filtering
        all_videos = get_all_videos()
        allowed_video_names = [video + ".mp4" for video in all_videos]
        
        # Adjust search parameters based on index type
        if self.index_type == "HIGH_ACCURACY":
            search_k = k  # No need to search more for flat index
        elif self.index_type == "HNSW":
            # Set search parameters for HNSW
            self.index.hnsw.efSearch = max(k * 2, 64)  # Dynamic ef based on k
            search_k = k
        else:
            search_k = min(k * 3, len(self.metadata))  # Search more to account for filtering
        
        # Perform search
        scores, indices = self.index.search(query_vector, search_k)
        
        # Filter and format results
        return_result = []
        scores_flat = scores[0]
        indices_flat = indices[0]
        
        for i, (score, idx) in enumerate(zip(scores_flat, indices_flat)):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
                
            if idx >= len(self.metadata):
                continue
                
            metadata = self.metadata[idx]
            video_name = metadata["video_name"]
            
            # Filter by video list if specified
            if video_name not in allowed_video_names:
                continue
            
            result = {
                "key": str(idx),
                "idx_folder": str(metadata["idx_folder"]),
                "video_name": video_name,
                "keyframe_id": str(metadata["frame_name"]).zfill(5),
                "fps": str(metadata["fps"]),
                "score": str(float(score))  # FAISS returns similarity scores
            }
            return_result.append(result)
            
            # Stop when we have enough results
            if len(return_result) >= k:
                break
        
        return return_result

    def deleteDatabase(self):
        """Delete the current index and metadata"""
        self.index = None
        self.metadata = []
        self.is_trained = False
        
        # Remove saved files
        index_path = self._get_index_path()
        metadata_path = self._get_metadata_path()
        
        try:
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            print("Database deleted successfully")
        except Exception as e:
            print(f"Error deleting database files: {e}")

    def getCount(self):
        """Get the total number of vectors in the index"""
        if self.index is None:
            return 0
        return self.index.ntotal 