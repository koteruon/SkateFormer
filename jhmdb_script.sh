#!/bin/sh

CONFIG_FILE="./config/train/jhmdb/SkateFormer_j.yaml"

i=86
while [ $i -lt 100 ]
do
  # 格式化 OUTPUT_DIR 為四位數 (0000 ~ 9999)
  OUTPUT_NUM=$(printf "%02d" $i)

  # 修改 OUTPUT_DIR 的最後數字
  sed -i "s|seed: .*|seed: ${i}|" "$CONFIG_FILE"
  sed -i "s|work_dir: .*|work_dir: ./work_dir/jhmdb/SkateFormer_j_2D_20250521_seed_${OUTPUT_NUM}/|" "$CONFIG_FILE"

  # 執行 python 指令，seed 直接使用數值格式
  python main.py --config "$CONFIG_FILE"

  # 增加計數
  i=$((i + 1))
done
