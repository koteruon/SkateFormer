import os

import cv2
import pandas as pd
from tqdm import tqdm

videos_path = "data/stroke_postures_pred/videos"
output_path = "data/stroke_postures_pred/output"
result_path = "work_dir/stroke_postures/SkateFormer_j_3D_20250711_02/runs-450-18900_top1f.csv"
intervals_path = "data/stroke_postures_pred/select_frame/M-3_left_01_annotations.csv"

os.makedirs(output_path, exist_ok=True)

stroke_id = {
    1: "Backhand Chop",
    2: "Backhand Flick",
    3: "Backhand Push",
    4: "Backhand Topspin",
    5: "Forehand Chop",
    6: "Forehand Drive",
    7: "Forehand Smash",
    8: "Forehand Topspin",
    9: "Background",
}

# 讀取預測結果
with open(result_path, "r") as file:
    lines = file.readlines()

csv_data = []
for line in lines:
    if line == "":
        break
    csv_data.append(line.split(","))

df = pd.DataFrame(
    csv_data,
    columns=[
        "movie_name_with_dir",
        "timestamp",
        "box_str_x1",
        "box_str_y1",
        "box_str_x2",
        "box_str_y2",
        "action_id",
        "score_str",
        "gt_action_id",
    ],
)
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# 設定文字顯示參數
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
color_score = (0, 235, 235)  # 黃色
thickness = 2
text_position = (40, 100)  # 顯示文字的位置

# 讀取有效的顯示區間
interval_df = pd.read_csv(intervals_path)

# 將每個區間轉換為 set（加速查找）
valid_frames = set()
for _, row in interval_df.iterrows():
    valid_frames.update(range(row["start_frame"], row["end_frame"] + 1))


videos_dir = os.listdir(videos_path)
for video_dir in videos_dir:
    frame_width = 2560
    frame_height = 1162
    fps = 60 // 4
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_file = os.path.join(output_path, video_dir + ".mp4")
    video_out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    movie_name_with_dir = os.path.join(videos_path, video_dir)
    file_names = sorted(os.listdir(movie_name_with_dir))
    started = False

    for file_name_with_extension in tqdm(file_names):
        file_path = os.path.join(movie_name_with_dir, file_name_with_extension)
        file_name, _ = os.path.splitext(file_name_with_extension)
        timestamp = str(int(file_name)).zfill(4)

        filtered_df = df[(df["movie_name_with_dir"].str.contains(video_dir)) & (df["timestamp"] == timestamp)]
        if not filtered_df.empty:
            started = True
        if not started:
            continue

        image = cv2.imread(file_path)

        if not filtered_df.empty:
            current_frame_number = int(file_name)
            if current_frame_number in valid_frames:
                action_id = filtered_df["action_id"].iloc[0]
                if int(action_id) != 9:
                    label_text = f"Stroke Type: {stroke_id[int(action_id)]}"
                    cv2.putText(image, label_text, text_position, font, font_scale, color_score, thickness, cv2.LINE_AA)

                video_out.write(image)

    video_out.release()
