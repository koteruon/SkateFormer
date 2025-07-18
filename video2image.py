import os

import cv2
from tqdm import tqdm


def video_to_images(video_path, output_folder):
    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)

    # 獲取影片檔案名（不包括副檔名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # 讀取影片
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"無法開啟影片: {video_path}")
        return

    # 獲取影片的總幀數
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 1
    with tqdm(total=total_frames, desc=f"Processing {video_name}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 保存圖片
            image_name = f"{frame_count:05d}.png"
            image_path = os.path.join(output_folder, image_name)
            cv2.imwrite(image_path, frame)

            frame_count += 1
            pbar.update(1)

    cap.release()
    print(f"影片 {video_path} 已轉換為 {frame_count} 張圖片並保存至 {output_folder}")


def convert_all_videos_in_folder(folder_path, output_folder):
    # 列出資料夾中的所有檔案
    files = os.listdir(folder_path)

    for video_file in files:
        video_path = os.path.join(folder_path, video_file)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        image_folder = os.path.join(output_folder, video_name)
        video_to_images(video_path, image_folder)


if __name__ == "__main__":
    input_folder = r"data/stroke_postures_pred/ori_videos"  # 更改為影片資料夾路徑
    output_folder = r"data/stroke_postures_pred/videos"  # 更改為保存圖片的資料夾路徑

    convert_all_videos_in_folder(input_folder, output_folder)
