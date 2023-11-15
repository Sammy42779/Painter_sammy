# !/bin/bash

set -x

JOB_NAME="painter_vit_large"
CKPT_FILE="painter_vit_large.pth"
PROMPT=ADE_train_00009574

SIZE=448
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"
CKPT_PATH="/hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth"

TASK=ade20k_segment

for EXP in Attack_VA Attack_ADA
do
    for EXP_ID in attack_AB attack_ABC
    do 
    # """ Baseline """
    OUT_PATH="/hhd3/ld/painter_sammy_output/transfer_eval/${TASK}/${EXP}/${EXP_ID}"
    DST_DIR="${OUT_PATH}/output/"
    SAVE_DATA_PATH="${OUT_PATH}/save_data/"


    # inference
    NUM_GPUS=3
    PORT=29522
    CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${PORT} --use_env \
      painter_inference_segm_transfer.py \
      --model ${MODEL} --prompt ${PROMPT} \
      --ckpt_path ${CKPT_PATH} --input_size ${SIZE} \
      --task ${TASK} \
      --exp ${EXP} \
      --exp_id ${EXP_ID} \
      --dst_dir ${DST_DIR} \
      --save_data_path ${SAVE_DATA_PATH} 

  done
done 




for EXP in Attack_VA Attack_ADA
do
    for EXP_ID in attack_AB attack_ABC
    do 
    # """ Baseline """
    OUT_PATH="/hhd3/ld/painter_sammy_output/transfer_eval/${TASK}/${EXP}"
    DST_DIR="${OUT_PATH}/output/"
    SAVE_DATA_PATH="${OUT_PATH}/save_data/"


    # postprocessing and eval
    CUDA_VISIBLE_DEVICES=0 python ADE20kSemSegEvaluatorCustom.py \
      --pred_dir ${DST_DIR}
  done
done 