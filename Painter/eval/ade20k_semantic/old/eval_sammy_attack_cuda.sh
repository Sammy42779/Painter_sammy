# !/bin/bash

# 2000 val images

# 单卡17176M, 8min左右
# 两卡17310M, 6min+
# 三卡17318M, 4min+

## attack 19066M





## PGD step=10 5卡 1h


set -x

NUM_GPUS=5
JOB_NAME="painter_vit_large"
CKPT_FILE="painter_vit_large.pth"
PROMPT=ADE_train_00009574

# ATTACK_ID=attack_A
# ATTACK_ID=attack_B
# ATTACK_ID=attack_C
ATTACK_ID=attack_AC
# ATTACK_ID=attack_AB
# ATTACK_ID=none

EPSILON=8

ATTACK=PGD
# ATTACK=FGSM
STEP=50


SIZE=448
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"

CKPT_PATH="/hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth"
# DST_DIR="models_inference/${JOB_NAME}/ade20k_semseg_inference_${CKPT_FILE}_${PROMPT}_size${SIZE}"
# DST_DIR="/hhd3/ld/data/ade20k/output_attack/${ATTACK}/${ATTACK_ID}_${EPSILON}"
DST_DIR="/hhd3/ld/data/ade20k/output_attack/${ATTACK}_${STEP}/${ATTACK_ID}_${EPSILON}"

# inference
CUDA_VISIBLE_DEVICES=1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=2000 --use_env \
  eval/ade20k_semantic/painter_inference_segm_attack_with_clip.py \
  --model ${MODEL} --prompt ${PROMPT} \
  --ckpt_path ${CKPT_PATH} --input_size ${SIZE} \
  --attack_id ${ATTACK_ID} \
  --epsilon ${EPSILON} \
  --attack_method ${ATTACK} \
  --num_steps ${STEP} 






# postprocessing and eval
# CUDA_VISIBLE_DEVICES=5 python eval/ade20k_semantic/ADE20kSemSegEvaluatorCustom.py \
#   --pred_dir ${DST_DIR}



# bash eval/ade20k_semantic/eval_sammy_attack_cuda.sh