# !/bin/bash

# 只能单卡跑，15706M 3min左右
# 654 test images
## 108服务器: /data1/; 110服务器: /hhd3/

set -x

JOB_NAME="painter_vit_large"  # 
CKPT_FILE="painter_vit_large.pth"
PROMPT="nyu_office_1/rgb_00009"

MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"
CKPT_PATH="/hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth"
DST_DIR="/hhd3/ld/data/nyu_depth_v2/component_${PROMPT}"

# inference
CUDA_VISIBLE_DEVICES=2 python eval/nyuv2_depth/painter_inference_depth_lesslike_AB_pair.py \
  --ckpt_path /hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth \
  --model ${MODEL} \
  --prompt ${PROMPT}

CUDA_VISIBLE_DEVICES=2 python eval/nyuv2_depth/eval_with_pngs.py \
  --pred_path ${DST_DIR} \
  --gt_path /hhd3/ld/data/nyu_depth_v2/official_splits/test/ \
  --dataset nyu --min_depth_eval 1e-3 --max_depth_eval 10 --eigen_crop


# bash eval/nyuv2_depth/eval_sammy_lesslike_AB_pair.sh 