# !/bin/bash

# 2000 val images

# 单卡17176M, 8min左右
# 两卡17310M, 6min+
# 三卡17318M, 4min+

# eval 单卡13680M, 

set -x

NUM_GPUS=4
JOB_NAME="painter_vit_large"
CKPT_FILE="painter_vit_large.pth"
PROMPT=ADE_train_00009574
# EXP_ID="exp_mapping_1_1_b"
# EXP_ID="exp_input_distribution_2_1_a"
# EXP_ID="exp_output_distribution_3_1_a"
# EXP_ID="exp_baseline"
# EXP_ID="exp_format_4_1_a"
# EXP_ID="exp_mapping_1_1_a"
# EXP_ID='exp_mapping_1_1_b_val'
# EXP_ID='exp_POS_A_mask_A'
# EXP_ID='exp_POS_B_mask_B'
# EXP_ID='exp_POS_AB_mask_AB'  # same as format_4_1_a
# EXP_ID='exp_POS_A_mask_A_white'
EXP_ID=exp_POS_A_ood_A_animeGAN
TRANSFER_IMG="/hhd3/ld/data/ade20k/AnimeGANv2/training/ADE_train_00009574_animeGAN.png"
# EXP_ID=exp_POS_B_ood_B_animeGAN
# TRANSFER_IMG="/hhd3/ld/data/ade20k/AnimeGANv2/validation/ADE_train_00009574_animeGAN_v1.png"

SIZE=448
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"

CKPT_PATH="/hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth"
# DST_DIR="models_inference/${JOB_NAME}/ade20k_semseg_inference_${CKPT_FILE}_${PROMPT}_size${SIZE}"
DST_DIR="/hhd3/ld/data/ade20k/ade20k_seg_inference_${CKPT_FILE}_${PROMPT}_${EXP_ID}"

# inference
CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=2000 --use_env \
  eval/ade20k_semantic/painter_inference_segm_exp_components.py \
  --model ${MODEL} --prompt ${PROMPT} \
  --ckpt_path ${CKPT_PATH} --input_size ${SIZE} \
  --exp_id ${EXP_ID} \
  --transfer_img ${TRANSFER_IMG}

# postprocessing and eval
CUDA_VISIBLE_DEVICES=1,2,3,4 python eval/ade20k_semantic/ADE20kSemSegEvaluatorCustom.py \
  --pred_dir ${DST_DIR}
