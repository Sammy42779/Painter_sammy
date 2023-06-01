# !/bin/bash

set -x

JOB_NAME="painter_vit_large"
CKPT_FILE="painter_vit_large.pth"
PROMPT=ADE_train_00009574

SIZE=448
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"
CKPT_PATH="/hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth"

TASK=ade20k_segment

# """ Demonstration """
STYLE_CHANGE_A="/hhd3/ld/data/ade20k/AnimeGANv2/training/ADE_train_00009574_animeGAN.png"
STYLE_CHANGE_B="/hhd3/ld/data/ade20k/AnimeGANv2/validation/ADE_train_00009574_animeGAN.png"

EXP=Demonstration


for EXP_ID in POS_A_mask_A POS_B_mask_B POS_AB_mask_AB POS_A_random_A_same_task POS_B_random_B_same_task POS_A_random_A_other_task POS_B_random_B_other_task POS_A_animeGAN_A POS_B_animeGAN_B POS_AB_animeGAN_AB
do 

OUT_PATH="/hhd3/ld/painter_sammy_output/${TASK}/${EXP}/${EXP_ID}"
DST_DIR="${OUT_PATH}/output/"
SAVE_DATA_PATH="${OUT_PATH}/save_data/"


# inference
NUM_GPUS=5
PORT=29504
CUDA_VISIBLE_DEVICES=2,5,6,7,8 python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${PORT} --use_env \
  painter_inference_segm.py \
  --model ${MODEL} --prompt ${PROMPT} \
  --ckpt_path ${CKPT_PATH} --input_size ${SIZE} \
  --task ${TASK} \
  --exp ${EXP} \
  --dst_dir ${DST_DIR} \
  --save_data_path ${SAVE_DATA_PATH} \
  --exp_id ${EXP_ID} \
  --style_change_A ${STYLE_CHANGE_A} \
  --style_change_B ${STYLE_CHANGE_B} \
  --save_demon

done 


for EXP_ID in POS_A_mask_A POS_B_mask_B POS_AB_mask_AB POS_A_random_A_same_task POS_B_random_B_same_task POS_A_random_A_other_task POS_B_random_B_other_task POS_A_animeGAN_A POS_B_animeGAN_B POS_AB_animeGAN_AB
do 

OUT_PATH="/hhd3/ld/painter_sammy_output/${TASK}/${EXP}/${EXP_ID}"
DST_DIR="${OUT_PATH}/output/"
SAVE_DATA_PATH="${OUT_PATH}/save_data/"

# postprocessing and eval
CUDA_VISIBLE_DEVICES=5 python ADE20kSemSegEvaluatorCustom.py \
  --pred_dir ${DST_DIR}

done 