# !/bin/bash

# 2000 val images

# 单卡17176M, 8min左右
# 两卡17310M, 6min+
# 三卡17318M, 4min+

# eval 单卡13680M, bash eval/ade20k_semantic/eval_sammy_exp_components.sh 

set -x


JOB_NAME="painter_vit_large"
CKPT_FILE="painter_vit_large.pth"
PROMPT=ADE_train_00009574


EXP_ID=baseline
# EXP_ID=POS_A_mask_A
# EXP_ID=POS_B_mask_B
# EXP_ID=POS_AB_mask_AB

# EXP_ID=POS_A_random_A_same_task
# EXP_ID=POS_A_random_A_other_task
# EXP_ID=POS_B_random_B_same_task
# EXP_ID=POS_B_random_B_other_task

# EXP_ID=POS_A_animeGAN_A
# EXP_ID=POS_B_animeGAN_B
# EXP_ID=POS_AB_animeGAN_AB

TRANSFER_IMG_A="/hhd3/ld/data/ade20k/AnimeGANv2/training/ADE_train_00009574_animeGAN.png"
TRANSFER_IMG_B="/hhd3/ld/data/ade20k/AnimeGANv2/validation/ADE_train_00009574_animeGAN.png"


SIZE=448
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"


CKPT_PATH="/hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth"


TASK=ade20k
EXP=component_analysis
DST_DIR="/hhd3/ld/painter_output/${TASK}/${EXP}/${EXP_ID}/output/"
SAVE_DATA_PATH="/hhd3/ld/painter_output/${TASK}/${EXP}/${EXP_ID}/save_data/"


NUM_GPUS=3
MASTER_PORT=1299

# inference
CUDA_VISIBLE_DEVICES=0,1,7 python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} --use_env \
  eval/ade20k_semantic/painter_inference_segm_exp_components.py \
  --model ${MODEL} --prompt ${PROMPT} \
  --ckpt_path ${CKPT_PATH} --input_size ${SIZE} \
  --exp_id ${EXP_ID} \
  --transfer_img_A ${TRANSFER_IMG_A} \
  --transfer_img_B ${TRANSFER_IMG_B} \
  --dst_dir ${DST_DIR} \
  --save_data_path ${SAVE_DATA_PATH}

# postprocessing and eval
CUDA_VISIBLE_DEVICES=0 python eval/ade20k_semantic/ADE20kSemSegEvaluatorCustom.py \
  --pred_dir ${DST_DIR}

# bash eval/ade20k_semantic/eval_sammy_exp_components.sh