
import os
from typing import Dict, List, Any
import cv2
import json
import numpy as np

import sys
sys.path.append("/workspace/competitions/AIC_2025/SIU_Sayan/Base/Split_Frame/Shot_Detection-main")

from shot_detecion_selector import ShotDetection
from io_setup import setup_video_path, SceneJsonLoader, CutKeyFrameLoader

model = ShotDetection('autoshot')

def split_video_into_frame(input_dir, output_folder):

    all_video_paths = setup_video_path(input_dir)


    prediction_scenes = model.run_model(video_path_dict=all_video_paths)

    sceneJson_dir = output_folder + "/SceneJson"
    os.makedirs(sceneJson_dir, exist_ok=True)
    json_handling = SceneJsonLoader(
        prediction_scenes,
        sceneJson_dir
    )
    json_handling.save_results()

    keyframe_dir = output_folder+ "/keyframe"
    keyframe_handler = CutKeyFrameLoader(
        sceneJson_dir,
        keyframe_dir
    )
    keyframe_handler.extract_keyframes(
        all_video_paths
    )

split_video_into_frame("/dataset/AIC_2025/SIU_Sayan/video",
                       "/dataset/AIC_2025/SIU_Sayan/autoshot")