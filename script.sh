#!/bin/sh

CONFIG_FILE="./config/train/table_tennis/SkateFormer_b.yaml"

i=3
while [ $i -lt 100 ]
do
  # 格式化 OUTPUT_DIR 為四位數 (0000 ~ 9999)
  OUTPUT_NUM=$(printf "%02d" $i)

  # 修改 OUTPUT_DIR 的最後數字
  sed -i "s|seed: .*|seed: ${i}|" "$CONFIG_FILE"
  sed -i "s|work_dir: .*|work_dir: ./work_dir/table_tennis/cs/SkateFormer_b_2D_20250310_${OUTPUT_NUM}/|" "$CONFIG_FILE"

  # 執行 python 指令，seed 直接使用數值格式
  python main.py --config "$CONFIG_FILE" --device 1

  # 增加計數
  i=$((i + 1))
done
