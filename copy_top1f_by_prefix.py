import os
import shutil

# 設定來源與目標路徑
src_root = "work_dir/jhmdb"
dst_root = "work_dir/top1f"

# 你想搜尋的錢墜（prefix）清單
prefix_list = ["SkateFormer_j_2D_20250521_", "SkateFormer_j_2D_20250531_"]  # 自行修改

# 遞迴搜尋 src_root 下所有子資料夾
for root, dirs, files in os.walk(src_root):
    for file in files:
        if file.endswith("top1f.csv"):
            rel_path = os.path.relpath(root, src_root)
            folder_name = os.path.basename(root)

            # 檢查是否符合 prefix
            if any(folder_name.startswith(prefix) for prefix in prefix_list):
                src_file_path = os.path.join(root, file)
                dst_dir_path = os.path.join(dst_root, rel_path)
                dst_file_path = os.path.join(dst_dir_path, file)

                # 建立目錄（如果尚未存在）
                os.makedirs(dst_dir_path, exist_ok=True)

                # 複製檔案
                shutil.copy2(src_file_path, dst_file_path)
                print(f"✅ 複製 {src_file_path} → {dst_file_path}")
