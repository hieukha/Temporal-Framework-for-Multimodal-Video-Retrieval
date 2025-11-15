import faiss
import numpy as np
import os
import ujson
import time
import pickle
from typing import List

DATASET_PATH_TEAM = "/dataset/AIC_2025/SIU_Sayan/keyframes"
DATASET_INDEX = "/dataset/AIC_2025/SIU_Sayan/autoshot/index"

dict_fps_0 = '/dataset/AIC_2025/SIU_Sayan/autoshot/video_fps_0.json'
with open(dict_fps_0, encoding='utf-8-sig') as json_file:
    dict_fps = ujson.load(json_file)


class FAISS_GPU_LLM2:
    """
    Memory-efficient FAISS helper for LLM2Clip (dim=1280):
    - Uses IVFPQ (FAST) by default to reduce GPU memory usage
    - Streams vectors in batches without keeping whole matrix in RAM/GPU
    - Trains IVF on a capped sample set
    """

    def __init__(self, collection_name: str = None, index_type: str = "FAST", temp_memory_gb: float = None):
        self.device = faiss.StandardGpuResources()
        self.collection_name = collection_name
        self.index_type = index_type
        self.temp_memory_gb = temp_memory_gb
        if temp_memory_gb is not None and temp_memory_gb > 0:
            self.device.setTempMemory(int(temp_memory_gb * 1024 * 1024 * 1024))
        self.index = None
        self.metadata = []
        self.is_trained = False
        self.size = None
        self.cache_dir = f"/workspace/competitions/AIC_2025/SIU_Sayan/Base/Cache/faiss_indices"
        os.makedirs(self.cache_dir, exist_ok=True)
        if collection_name:
            self._load_index()

    def _get_index_path(self):
        return os.path.join(self.cache_dir, f"{self.collection_name}_{self.index_type}.index")

    def _get_metadata_path(self):
        return os.path.join(self.cache_dir, f"{self.collection_name}_{self.index_type}_metadata.pkl")

    def _create_index(self, feature_size: int):
        self.size = feature_size
        if self.index_type == "FAST":
            nlist = 4096
            m = 64
            pq_bits = 8
            cpu_index = faiss.IndexIVFPQ(
                faiss.IndexFlatIP(feature_size), feature_size, nlist, m, pq_bits, faiss.METRIC_INNER_PRODUCT
            )
        elif self.index_type == "BALANCED":
            nlist = 8192
            cpu_index = faiss.IndexIVFFlat(
                faiss.IndexFlatIP(feature_size), feature_size, nlist, faiss.METRIC_INNER_PRODUCT
            )
        elif self.index_type in ("HIGH_ACCURACY", "HIGH_ACCURACY_FP16"):
            # Use GPU Flat index directly, with optional FP16 storage to reduce memory
            try:
                cfg = faiss.GpuIndexFlatConfig()
                cfg.device = 0
                cfg.useFloat16 = (self.index_type == "HIGH_ACCURACY_FP16")
                self.index = faiss.GpuIndexFlatIP(self.device, feature_size, cfg)
                return self.index
            except Exception:
                # Fallback to CPU Flat
                cpu_index = faiss.IndexFlatIP(feature_size)
                self.index = cpu_index
                return self.index
        else:
            # Fallback to CPU Flat for unknown types
            cpu_index = faiss.IndexFlatIP(feature_size)
            self.index = cpu_index
            return self.index

        # Move to GPU for IVF-based indexes
        try:
            self.index = faiss.index_cpu_to_gpu(self.device, 0, cpu_index)
        except Exception:
            self.index = cpu_index
        return self.index

    def _load_index(self):
        index_path = self._get_index_path()
        metadata_path = self._get_metadata_path()
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                cpu_index = faiss.read_index(index_path)
                try:
                    self.index = faiss.index_cpu_to_gpu(self.device, 0, cpu_index)
                except Exception:
                    self.index = cpu_index
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                if self.metadata:
                    if 'feature_size' in self.metadata[0]:
                        self.size = self.metadata[0]['feature_size']
                self.is_trained = True
                print(f"Successfully loaded {self.index_type} index with {len(self.metadata)} vectors")
                return True
            except Exception as e:
                print(f"Error loading index: {e}")
        return False

    def _save_index(self):
        index_path = self._get_index_path()
        metadata_path = self._get_metadata_path()
        try:
            if hasattr(self.index, 'getDevice'):
                cpu_index = faiss.index_gpu_to_cpu(self.index)
            else:
                cpu_index = self.index
            faiss.write_index(cpu_index, index_path)
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

    def _iter_feature_files(self, FEATURES_PATH: List[str]):
        for idx_folder, folder_path in enumerate(FEATURES_PATH):
            if not os.path.exists(folder_path):
                continue
            for feat_npy in sorted(os.listdir(folder_path)):
                if not feat_npy.endswith('.npy'):
                    continue
                yield idx_folder, folder_path, feat_npy

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def _collect_train_samples(self, feature_size: int, FEATURES_PATH: List[str], max_samples: int = 300000):
        """Collect up to max_samples vectors for IVF training without loading everything."""
        samples = []
        per_file_cap = 200  # take up to 200 frames per file to diversify
        for _, folder_path, feat_npy in self._iter_feature_files(FEATURES_PATH):
            feats_arr = np.load(os.path.join(folder_path, feat_npy))
            take = min(per_file_cap, len(feats_arr))
            if take <= 0:
                continue
            subset = feats_arr[:take]
            for feat in subset:
                if hasattr(feat, 'cpu'):
                    feat = feat.cpu().detach().numpy()
                if not isinstance(feat, np.ndarray):
                    feat = np.array(feat)
                feat = feat.astype(np.float32).reshape(-1)
                if feat.shape[0] != feature_size:
                    continue
                samples.append(self._normalize(feat))
                if len(samples) >= max_samples:
                    break
            if len(samples) >= max_samples:
                break
        if len(samples) == 0:
            return None
        return np.array(samples, dtype=np.float32)

    def addDatabase(self, collection_name, feature_size, FEATURES_PATH, SPLIT_NAME):
        self.collection_name = collection_name
        self.size = feature_size
        self._create_index(feature_size)

        print("Starting streaming add for LLM2Clip (memory efficient)...")

        # Train IVF if needed
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            print("Collecting training samples...")
            train_vectors = self._collect_train_samples(feature_size, FEATURES_PATH)
            if train_vectors is None or len(train_vectors) < 1000:
                raise RuntimeError("Not enough samples to train IVF index")
            print(f"Training with {len(train_vectors):,} samples")
            self.index.train(train_vectors)
            print("Training completed!")

        self.metadata = []
        total_vectors = 0
        start_time = time.time()

        batch: List[np.ndarray] = []
        batch_meta: List[dict] = []
        batch_size = 20000  # stream in moderate batches

        for idx_folder, folder_path, feat_npy in self._iter_feature_files(FEATURES_PATH):
            video_name = feat_npy.split('.')[0]
            video_name_no_ext = video_name.split(".")[0]
            frame_path = os.path.join(DATASET_PATH_TEAM, f"Keyframes_{video_name_no_ext.split('_')[0]}", "keyframes", video_name_no_ext)
            if not os.path.exists(frame_path):
                print(f"Warning: Frame path not found for {video_name_no_ext} at {frame_path}")
                continue
            frame_list = sorted(os.listdir(frame_path))

            feats_arr = np.load(os.path.join(folder_path, feat_npy))
            for idx, feat in enumerate(feats_arr):
                if hasattr(feat, 'cpu'):
                    feat = feat.cpu().detach().numpy()
                if not isinstance(feat, np.ndarray):
                    feat = np.array(feat)
                feat_np = feat.astype(np.float32).reshape(-1)
                if feat_np.shape[0] != feature_size:
                    continue
                feat_np = self._normalize(feat_np)
                batch.append(feat_np)
                batch_meta.append({
                    "idx_folder": idx_folder,
                    "video_name": video_name + ".mp4",
                    "frame_name": frame_list[idx].replace(".jpg",""),
                    "fps": dict_fps.get(video_name, 25)
                })
                if len(batch) >= batch_size:
                    arr = np.array(batch, dtype=np.float32)
                    self.index.add(arr)
                    self.metadata.extend(batch_meta)
                    total_vectors += len(batch)
                    print(f"Added {total_vectors:,} vectors...")
                    batch = []
                    batch_meta = []

        if batch:
            arr = np.array(batch, dtype=np.float32)
            self.index.add(arr)
            self.metadata.extend(batch_meta)
            total_vectors += len(batch)

        self.is_trained = True
        total_time = time.time() - start_time
        print(f"Completed streaming add: {total_vectors:,} vectors in {total_time:.2f}s")
        self._save_index()
        return {"status": "success", "total_vectors": total_vectors, "index_type": self.index_type}

    def search(self, query, k, video_list=None):
        if not self.is_trained or self.index is None:
            raise ValueError("Index is not trained or loaded. Please run addDatabase first.")
        q = query / np.linalg.norm(query)
        q = q.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append({
                "key": str(idx),
                "idx_folder": str(meta["idx_folder"]),
                "video_name": meta["video_name"],
                "keyframe_id": str(meta["frame_name"]).zfill(5),
                "fps": str(meta["fps"]),
                "score": str(float(score))
            })
            if len(results) >= k:
                break
        return results

    def getCount(self):
        if self.index is None:
            return 0
        return self.index.ntotal


