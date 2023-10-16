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
EPSILON=2
STEPS=10

# """ Attack_VA, Attack_AA, Attack_DA, Attack_ADA """

EXP=Attack_AA

# for MASK_RATIO in 0.1
# do
#     for EXP_ID in attack_C attack_AB attack_ABC
#     do
#         for LAM_AB in 0.1 0.01 0.001
#         do 

#         OUT_PATH="/hhd3/ld/painter_sammy_output/${TASK}/${EXP}/${ATTACK_METHOD}_eps${EPSILON}_steps${STEPS}/ignore_D_loss/mask_ratio_${MASK_RATIO}/lamAB_${LAM_AB}/${EXP_ID}"
#         DST_DIR="${OUT_PATH}/output/"
#         SAVE_DATA_PATH="${OUT_PATH}/save_data/"


#         # inference
#         NUM_GPUS=4
#         PORT=29504
#         CUDA_VISIBLE_DEVICES=7,8,5,6 python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${PORT} --use_env \
#           painter_inference_segm.py \
#           --model ${MODEL} --prompt ${PROMPT} \
#           --ckpt_path ${CKPT_PATH} --input_size ${SIZE} \
#           --task ${TASK} \
#           --exp ${EXP} \
#           --dst_dir ${DST_DIR} \
#           --save_data_path ${SAVE_DATA_PATH} \
#           --exp_id ${EXP_ID} \
#           --attack_method ${ATTACK_METHOD} \
#           --epsilon ${EPSILON} \
#           --num_steps ${STEPS} \
#           --mask_B \
#           --ignore_D_loss \
#           --lam_AB ${LAM_AB} \
#           --mask_ratio ${MASK_RATIO} \
#           --save_adv
#         done 
#     done 
# done           


for MASK_RATIO in 0.75 0.5 0.25 0.1
do
    for EXP_ID in attack_C attack_AB attack_ABC
    do
        for LAM_AB in 0.1 0.01 0.001
        do 
        
        OUT_PATH="/hhd3/ld/painter_sammy_output/${TASK}/${EXP}/${ATTACK_METHOD}_eps${EPSILON}_steps${STEPS}/ignore_D_loss/mask_ratio_${MASK_RATIO}/lamAB_${LAM_AB}/${EXP_ID}"
        DST_DIR="${OUT_PATH}/output/"
        SAVE_DATA_PATH="${OUT_PATH}/save_data/"



      # postprocessing and eval
      CUDA_VISIBLE_DEVICES=6 python ADE20kSemSegEvaluatorCustom.py \
        --pred_dir ${DST_DIR}
        done 
    done 
done           
