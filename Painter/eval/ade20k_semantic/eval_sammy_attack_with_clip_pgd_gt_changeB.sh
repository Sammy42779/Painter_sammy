# !/bin/bash

# 2000 val images

# 单卡17176M, 8min左右
# 两卡17310M, 6min+
# 三卡17318M, 4min+

## attack 19066M


## PGD step=10 5卡 1h



# ATTACK_ID=attack_A
# ATTACK_ID=attack_B
# ATTACK_ID=attack_C
# ATTACK_ID=attack_AB
# ATTACK_ID=attack_AC
# ATTACK_ID=attack_BC
# ATTACK_ID=attack_ABC
# ATTACK_ID=none

for ATTACK_ID in attack_A attack_B attack_C attack_AB attack_AC attack_BC attack_ABC none

do

  set -x

  JOB_NAME="painter_vit_large"
  CKPT_FILE="painter_vit_large.pth"
  PROMPT=ADE_train_00009574


  EPSILON=8
  ATTACK=PGD
  STEP=10

  SIZE=448
  MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"

  CKPT_PATH="/hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth"

  TASK=ade20k
  EXP=attack_baseline_changeB
  DST_DIR="/hhd3/ld/painter_output/${TASK}/${EXP}/${ATTACK}${STEP}_${EPSILON}/${ATTACK_ID}/output/"
  SAVE_DATA_PATH="/hhd3/ld/painter_output/${TASK}/${EXP}/${ATTACK}${STEP}_${EPSILON}/${ATTACK_ID}/save_data/"


  NUM_GPUS=5
  # inference
  CUDA_VISIBLE_DEVICES=0,1,2,3,5 python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=1997 --use_env \
    eval/ade20k_semantic/painter_inference_segm_attack_gt_baseline_changeB.py \
    --model ${MODEL} --prompt ${PROMPT} \
    --ckpt_path ${CKPT_PATH} --input_size ${SIZE} \
    --attack_id ${ATTACK_ID} \
    --epsilon ${EPSILON} \
    --attack_method ${ATTACK} \
    --num_steps ${STEP} \
    --dst_dir ${DST_DIR} \
    --save_data_path ${SAVE_DATA_PATH}


  # postprocessing and eval
  CUDA_VISIBLE_DEVICES=3 python eval/ade20k_semantic/ADE20kSemSegEvaluatorCustom.py \
    --pred_dir ${DST_DIR}

done

# bash eval/ade20k_semantic/eval_sammy_attack_with_clip_pgd_gt_changeB.sh