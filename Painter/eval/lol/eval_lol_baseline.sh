TASK=lol_enhance

# """ Baseline """
EXP=Baseline
OUT_PATH="/hhd3/ld/painter_sammy_output/${TASK}/${EXP}"
DST_DIR="${OUT_PATH}/output/"
SAVE_DATA_PATH="${OUT_PATH}/save_data/"

CUDA_VISIBLE_DEVICES=6 python painter_inference_lol.py \
    --ckpt_path /hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth \
    --task ${TASK} \
    --exp ${EXP} \
    --dst_dir ${DST_DIR} \
    --save_data_path ${SAVE_DATA_PATH} 





