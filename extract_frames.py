import os

import cv2

# 來源與輸出資料夾
input_dir = "data/stroke_postures/ori_video"
output_dir = "data/stroke_postures/video"

# 確保輸出資料夾存在
os.makedirs(output_dir, exist_ok=True)

# 處理每個影片
for filename in os.listdir(input_dir):
    if not filename.endswith(".mp4"):
        continue

    video_path = os.path.join(input_dir, filename)
    video_name = os.path.splitext(filename)[0]  # 例如 forehand_chop_01

    # 建立對應的資料夾
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    # 讀取影片
    cap = cv2.VideoCapture(video_path)
    frame_index = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 檔名格式為00001.png
        filename = f"{frame_index:05d}.png"
        frame_path = os.path.join(video_output_dir, filename)
        cv2.imwrite(frame_path, frame)

        frame_index += 1

    cap.release()
    print(f"{video_name} done, total {frame_index-1} frames.")
