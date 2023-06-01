# !/bin/bash

set -x

JOB_NAME="painter_vit_large"
CKPT_FILE="painter_vit_large.pth"
PROMPT=ADE_train_00009574

SIZE=448
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"
CKPT_PATH="/hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth"

TASK=ade20k_segment

ATTACK_METHOD=PGD
EPSILON=8
STEPS=10


# """ Attack_VA, Attack_AA, Attack_DA, Attack_ADA """


## Attack_VA vanilla
EXP=Attack_VA

for EXP_ID in attack_A attack_B attack_C attack_AB attack_AC attack_BC attack_ABC
do 

OUT_PATH="/hhd3/ld/painter_sammy_output/${TASK}/${EXP}/${EXP_ID}/${ATTACK_METHOD}_eps${EPSILON}_steps${STEPS}"
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
  --save_data_path ${SAVE_DATA_PATH}  \
  --exp_id ${EXP_ID} \
  --attack_method ${ATTACK_METHOD} \
  --epsilon ${EPSILON} \
  --num_steps ${STEPS} \
  --save_adv 

done 


for EXP_ID in attack_A attack_B attack_C attack_AB attack_AC attack_BC attack_ABC
do 

OUT_PATH="/hhd3/ld/painter_sammy_output/${TASK}/${EXP}/${EXP_ID}/${ATTACK_METHOD}_eps${EPSILON}_steps${STEPS}"
DST_DIR="${OUT_PATH}/output/"
SAVE_DATA_PATH="${OUT_PATH}/save_data/"

# postprocessing and eval
CUDA_VISIBLE_DEVICES=5 python ADE20kSemSegEvaluatorCustom.py \
  --pred_dir ${DST_DIR}
done 