import json
import os
from pathlib import Path

import cv2
import pandas as pd
import torch
from ultralytics import YOLO

action_map = {
    "bhc": "0",
    "bhpull": "1",
    "bhpush": "2",
    "bht": "3",
    "fhc": "4",
    "fhpull": "5",
    "fhpush": "6",
    "fhs": "7",
}

# 設定影像寬度
image_width = 1920

# 假設您有一個YOLOv11的姿勢估計模型
pose_model = YOLO("weights/yolo11x-pose.pt")

current_directory = os.getcwd()

data_path = os.path.join(current_directory, "data", "table_tennis_2D")

folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]

for folder in folders:
    folder_path = os.path.join(data_path, folder)

    # 設定輸出目錄
    output_dir = os.path.join(folder_path, "annotation")

    # 讀取CSV文件
    annotations = pd.read_csv(os.path.join(folder_path, f"{folder}_annotations.csv"))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 讀取影片
    video_path = os.path.join(folder_path, f"{folder}.mp4")
    cap = cv2.VideoCapture(video_path)

    # 處理每一段
    for idx, row in annotations.iterrows():
        start_frame = row["start_frame"]
        end_frame = row["end_frame"]

        # 設定影片讀取的起始與結束位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        clip_skeletons = []
        clip_bbox = []

        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            # 使用YOLOv11進行姿勢估計
            res = pose_model(frame)

            keypoints = [[0.0, 0.0] for _ in range(17)]
            bbox = [0.0, 0.0, 0.0, 0.0]

            res = res[0]
            bboxs = res.boxes.xyxy
            confs = res.boxes.conf
            if len(bboxs) == 0:
                clip_skeletons.append(keypoint)
                clip_bbox.append(bbox)
                continue

            person_index = torch.where(bboxs[:, 2] <= image_width / 2)[0]
            if len(person_index) == 0:
                clip_skeletons.append(keypoint)
                clip_bbox.append(bbox)
                continue

            person_index = torch.argmax(confs[person_index])

            assert person_index.numel() == 1

            keypoint = res.keypoints.xyn[person_index].squeeze(0).tolist()
            clip_skeletons.append(keypoint)

            bbox = bboxs[person_index].squeeze(0).tolist()
            clip_bbox.append(bbox)

        action_name = folder.split("_")[0]
        # 儲存為JSON
        json_data = {
            "file_name": f"{folder}_{idx:04d}",
            "skeletons": clip_skeletons,
            "label": action_map[action_name],
            "length": len(clip_skeletons),
            "bbox": clip_bbox,
        }

        # 儲存為JSON文件
        json_file = os.path.join(output_dir, f"{folder}_{idx:04d}.json")
        with open(json_file, "w") as f:
            json.dump(json_data, f, indent=4)

# 釋放資源
cap.release()
