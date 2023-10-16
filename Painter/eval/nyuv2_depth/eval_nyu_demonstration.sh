# !/bin/bash

set -x

JOB_NAME="painter_vit_large"
CKPT_FILE="painter_vit_large.pth"
PROMPT="study_room_0005b/rgb_00094"

MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"
CKPT_PATH="/hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth"


TASK=nyu_depth

# """ Demonstration """
STYLE_CHANGE_A="/hhd3/ld/data/nyu_depth_v2/AnimeGANv2/rgb_00094_animeGAN.png"
STYLE_CHANGE_B="/hhd3/ld/data/nyu_depth_v2/AnimeGANv2/sync_depth_00094_animeGAN.png"

EXP=Demonstration


# for EXP_ID in POS_A_mask_A POS_B_mask_B POS_AB_mask_AB POS_A_random_A_same_task POS_B_random_B_same_task POS_A_random_A_other_task POS_B_random_B_other_task POS_A_animeGAN_A POS_B_animeGAN_B POS_AB_animeGAN_AB
for EXP_ID in POS_AB_animeGAN_AB
do 

OUT_PATH="/hhd3/ld/painter_sammy_output/${TASK}/${EXP}/${EXP_ID}"
DST_DIR="${OUT_PATH}/output/"
SAVE_DATA_PATH="${OUT_PATH}/save_data/"

# inference
CUDA_VISIBLE_DEVICES=1 python painter_inference_depth.py \
  --ckpt_path ${CKPT_PATH} \
  --model ${MODEL} \
  --prompt ${PROMPT} \
  --task ${TASK} \
  --exp ${EXP} \
  --dst_dir ${DST_DIR} \
  --save_data_path ${SAVE_DATA_PATH} \
  --exp_id ${EXP_ID} \
  --style_change_A ${STYLE_CHANGE_A} \
  --style_change_B ${STYLE_CHANGE_B} \
  --save_demon

CUDA_VISIBLE_DEVICES=1 python eval_with_pngs.py \
  --pred_path ${DST_DIR} \
  --gt_path /hhd3/ld/data/nyu_depth_v2/official_splits/test/ \
  --dataset nyu --min_depth_eval 1e-3 --max_depth_eval 10 --eigen_crop

done