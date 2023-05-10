# !/bin/bash

# 2000 val images

# 单卡17176M, 8min左右
# 两卡17310M, 6min+
# 三卡17318M, 4min+

# eval 单卡13680M, bash eval/ade20k_semantic/eval_sammy_exp_components.sh 

set -x

NUM_GPUS=4
JOB_NAME="painter_vit_large"
CKPT_FILE="painter_vit_large.pth"
PROMPT=ADE_train_00009574

MASTER_PORT=1299


# EXP_ID=baseline

# EXP_ID=exp_POS_A_mask_A
# EXP_ID=exp_POS_B_mask_B
# EXP_ID=exp_POS_AB_mask_AB  # same as format_4_1_a
# EXP_ID=exp_POS_AB_exchange_AB

# EXP_ID=exp_POS_A_random_A_same_task
# EXP_ID=exp_POS_A_random_A_other_task
# EXP_ID=exp_POS_B_random_B_same_task
# EXP_ID=POS_B_random_B_other_task

# EXP_ID=POS_B_random_B_other_task_Flickr
# EXP_ID=exp_POS_B_random_B_other_task_random
# TRANSFER_IMG="placeholder"

# EXP_ID=exp_POS_A_random_A_other_task_coco_gt

# EXP_ID=exp_POS_A_animeGAN_A
# TRANSFER_IMG="/hhd3/ld/data/ade20k/AnimeGANv2/training/ADE_train_00009574_animeGAN.png"
EXP_ID=POS_B_animeGAN_B
TRANSFER_IMG="/hhd3/ld/data/ade20k/AnimeGANv2/validation/ADE_train_00009574_animeGAN.png"


SIZE=448
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"

CKPT_PATH="/hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth"
# DST_DIR="models_inference/${JOB_NAME}/ade20k_semseg_inference_${CKPT_FILE}_${PROMPT}_size${SIZE}"
DST_DIR="/hhd3/ld/data/ade20k/output/ade20k_seg_inference_${CKPT_FILE}_${PROMPT}_${EXP_ID}"

# inference
CUDA_VISIBLE_DEVICES=0,1,4,5 python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} --use_env \
  eval/ade20k_semantic/painter_inference_segm_exp_components.py \
  --model ${MODEL} --prompt ${PROMPT} \
  --ckpt_path ${CKPT_PATH} --input_size ${SIZE} \
  --exp_id ${EXP_ID} \
  --transfer_img ${TRANSFER_IMG}

# postprocessing and eval
CUDA_VISIBLE_DEVICES=4 python eval/ade20k_semantic/ADE20kSemSegEvaluatorCustom.py \
  --pred_dir ${DST_DIR}
