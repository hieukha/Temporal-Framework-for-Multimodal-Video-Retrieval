import sys
import os
import ujson
import numpy as np
import torch
import pathlib
from pathlib import Path
from PIL import Image


process_path = "/dataset/AIC_2025/SIU_Sayan/autoshot/keyframes/"

dict_duplicate = {}

for image in Path(process_path).glob("**/*.jpg"):
    if not image.is_file():
        continue
    image = str(image)
    json_folder = Path(image.replace("/frames/autoshot/","/duplicate_json/")).parents[0]
    json_folder.mkdir(parents=True, exist_ok=True)    
    json_name = str(image).replace("/frames/autoshot/","/duplicate_json/").replace(".jpg",".json")

    with open(json_name, 'r') as open_json:
        list_dup = ujson.load(open_json)

    parts = json_name.split('/')
    video_name = parts[-2]
    frame_name = parts[-1].split('.')[0]
        
    if len(list_dup)>1:
        if video_name not in dict_duplicate:
            dict_duplicate[video_name] = {}
        dict_duplicate[video_name][frame_name] = 1
    else:
        if video_name not in dict_duplicate:
            dict_duplicate[video_name] = {}        
        dict_duplicate[video_name][frame_name] = 0
    print(json_name)

with open('/dataset/AIC_2025/SIU_Sayan/duplicate_0_all.json','w') as save_duplicate:
    ujson.dump(dict_duplicate, save_duplicate, indent=4)