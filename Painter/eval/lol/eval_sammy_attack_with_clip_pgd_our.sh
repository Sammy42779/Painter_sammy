



# ATTACK_ID=attack_A
# ATTACK_ID=attack_B
# ATTACK_ID=attack_C
ATTACK_ID=attack_AB
# ATTACK_ID=attack_AC
# ATTACK_ID=attack_BC
# ATTACK_ID=attack_ABC
# ATTACK_ID=none
EPSILON=8
# ALPHA=128

ATTACK=PGD
STEP=10


TASK=light_enhance
SAVE_TASK=lol_enhance

METHOD=our_attack

PY_FILE=painter_inference_lol_attack_with_clip_our

#### PGD10步 2mins
DST_DIR="/hhd3/ld/data/${TASK}/${METHOD}/${ATTACK}${STEP}_${EPSILON}/${ATTACK_ID}/"
SAVE_DATA_PATH="/hhd3/ld/data/Painter_root/${SAVE_TASK}/${METHOD}/${ATTACK}${STEP}_${EPSILON}/${ATTACK_ID}/"


CUDA_VISIBLE_DEVICES=0 python eval/lol/${PY_FILE}.py \
    --ckpt_path /hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth \
    --attack_id ${ATTACK_ID} \
    --epsilon ${EPSILON} \
    --attack_method ${ATTACK} \
    --num_steps ${STEP} \
    --dst_dir ${DST_DIR} \
    --save_data_path ${SAVE_DATA_PATH}



## bash eval/lol/eval_sammy_attack_with_clip_pgd_our.sh