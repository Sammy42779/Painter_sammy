# !/bin/bash

# 只能单卡跑，15706M
# 654 test images
## 108服务器: /data1/; 110服务器: /hhd3/


# 1. input-ouput mapping的重要性
#   1.1 将图B替换为其他图片其他任务的ground-truth图片, 保证ID但是任务关系不对
#   1.2 将图B替换为其他图片同一任务的ground-truth图片, 保证ID但是gt和prompt不pair
# 2. input distribution的重要性
# 3. output distribution的重要性
# 4. format的重要性


# bash eval/nyuv2_depth/eval_sammy_exp_components.sh 

set -x

JOB_NAME="painter_vit_large"  # 
CKPT_FILE="painter_vit_large.pth"
PROMPT="study_room_0005b/rgb_00094"  # official prompt

# EXP_ID=baseline
# EXP_ID=exp_POS_A_mask_A
# EXP_ID='exp_POS_B_mask_B'
# EXP_ID='exp_POS_AB_mask_AB'  # same as format_4_1_a
# EXP_ID=exp_POS_A_ood_A_animeGAN
# EXP_ID=exp_POS_B_ood_B_animeGAN
# EXP_ID=exp_POS_AB_exchange_AB
# EXP_ID=exp_POS_B_random_B_same_task
# EXP_ID=POS_B_random_B_other_task
# EXP_ID=exp_POS_A_random_A_same_task
# EXP_ID=exp_POS_A_random_A_other_task
EXP_ID=exp_POS_A_random_A_other_task_coco_gt
# EXP_ID=POS_B_random_B_other_task_Flickr
# EXP_ID=exp_POS_B_random_B_other_task_random
TRANSFER_IMG="placeholder"

# EXP_ID=exp_POS_A_animeGAN_A
# TRANSFER_IMG="/hhd3/ld/data/nyu_depth_v2/AnimeGANv2/rgb_00094_animeGAN.png"
# EXP_ID=exp_POS_B_animeGAN_B
# TRANSFER_IMG="/hhd3/ld/data/nyu_depth_v2/AnimeGANv2/sync_depth_00094_animeGAN.png"



MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"
CKPT_PATH="/hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth"
DST_DIR="/hhd3/ld/data/nyu_depth_v2/component_analysis/nyuv2_depth_inference_${CKPT_FILE}_${PROMPT}_${EXP_ID}"

# inference
CUDA_VISIBLE_DEVICES=0 python eval/nyuv2_depth/painter_inference_depth_exp_components.py \
  --ckpt_path /hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth \
  --model ${MODEL} \
  --prompt ${PROMPT} \
  --exp_id ${EXP_ID} \
  --transfer_img ${TRANSFER_IMG}

CUDA_VISIBLE_DEVICES=0 python eval/nyuv2_depth/eval_with_pngs.py \
  --pred_path ${DST_DIR} \
  --gt_path /hhd3/ld/data/nyu_depth_v2/official_splits/test/ \
  --dataset nyu --min_depth_eval 1e-3 --max_depth_eval 10 --eigen_crop
