# !/bin/bash

set -x

JOB_NAME="painter_vit_large"
CKPT_FILE="painter_vit_large.pth"
PROMPT="study_room_0005b/rgb_00094"

MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"
CKPT_PATH="/hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth"

TASK=nyu_depth

ATTACK_METHOD=PGD
EPSILON=8
STEPS=10

# """ Attack_VA, Attack_AA, Attack_DA, Attack_ADA """


## Attack_DA distribution attack
EXP=Attack_DA

# for EXP_ID in attack_A attack_B attack_C attack_AB attack_AC attack_BC attack_ABC
for EXP_ID in attack_C attack_AB attack_ABC
do
    for LAM_AC in 0.1 0.01 0.001
    do

    OUT_PATH="/hhd3/ld/painter_sammy_output/${TASK}/${EXP}/${ATTACK_METHOD}_eps${EPSILON}_steps${STEPS}/lamAC_${LAM_AC}/${EXP_ID}"
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
      --attack_method ${ATTACK_METHOD} \
      --epsilon ${EPSILON} \
      --num_steps ${STEPS} \
      --break_AC \
      --lam_AC ${LAM_AC} \
      --with_B \
      --save_adv 

    CUDA_VISIBLE_DEVICES=1 python eval_with_pngs.py \
      --pred_path ${DST_DIR} \
      --gt_path /hhd3/ld/data/nyu_depth_v2/official_splits/test/ \
      --dataset nyu --min_depth_eval 1e-3 --max_depth_eval 10 --eigen_crop
    done 
done 