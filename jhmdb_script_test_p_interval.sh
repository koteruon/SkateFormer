#!/bin/sh

CONFIG_FILE="./config/train/jhmdb/SkateFormer_b.yaml"

for intra_p in $(seq 0.50 0.05 1.00)
do
  # 格式化 OUTPUT_NUM (e.g., 0.50 -> 050, 0.55 -> 055, ..., 1.00 -> 100)
  OUTPUT_NUM=$(printf "%03d" $(echo "$intra_p * 100" | bc | cut -d. -f1))

  # 修改 work_dir
  sed -i "s|work_dir: .*|work_dir: ./work_dir/jhmdb/SkateFormer_b_2D_20250320_${OUTPUT_NUM}/|" "$CONFIG_FILE"

  # 修改 intra_p
  sed -i "s|  p_interval: .*|  p_interval: [${intra_p}, 1]|" "$CONFIG_FILE"

  # 執行 Python 指令
  python main.py --config "$CONFIG_FILE"
done