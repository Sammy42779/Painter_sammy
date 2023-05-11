# !/bin/bash

# 2000 val images

# 单卡17176M, 8min左右
# 两卡17310M, 6min+
# 三卡17318M, 4min+

set -x

NUM_GPUS=3
JOB_NAME="painter_vit_large"
CKPT_FILE="painter_vit_large.pth"
PROMPT=ADE_train_00009574


SIZE=448
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"

CKPT_PATH="/hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth"
# DST_DIR="models_inference/${JOB_NAME}/ade20k_semseg_inference_${CKPT_FILE}_${PROMPT}_size${SIZE}"
DST_DIR="/hhd3/ld/data/ade20k/ade20k_seg_inference_${CKPT_FILE}_${PROMPT}"

# inference
CUDA_VISIBLE_DEVICES=0,1,4 python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=2000 --use_env \
  eval/ade20k_semantic/painter_inference_segm.py \
  --model ${MODEL} --prompt ${PROMPT} \
  --ckpt_path ${CKPT_PATH} --input_size ${SIZE}

# postprocessing and eval
CUDA_VISIBLE_DEVICES=1,3,4 python eval/ade20k_semantic/ADE20kSemSegEvaluatorCustom.py \
  --pred_dir ${DST_DIR}
