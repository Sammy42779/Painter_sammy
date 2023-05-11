



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

EPSILON=8
ATTACK=PGD
STEP=10

#### PGD10步 2mins

TASK=lol_enhance
EXP=attack_baseline_changeA
DST_DIR="/hhd3/ld/painter_output/${TASK}/${EXP}/${ATTACK}${STEP}_${EPSILON}/${ATTACK_ID}/output/"
SAVE_DATA_PATH="/hhd3/ld/painter_output/${TASK}/${EXP}/${ATTACK}${STEP}_${EPSILON}/${ATTACK_ID}/save_data/"


CUDA_VISIBLE_DEVICES=3 python eval/lol/painter_inference_lol_attack_gt_baseline_changeA.py \
    --ckpt_path /hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth \
    --attack_id ${ATTACK_ID} \
    --epsilon ${EPSILON} \
    --attack_method ${ATTACK} \
    --num_steps ${STEP} \
    --dst_dir ${DST_DIR} \
    --save_data_path ${SAVE_DATA_PATH}

done 

## bash eval/lol/eval_sammy_attack_with_clip_pgd_gt_changeA.sh

