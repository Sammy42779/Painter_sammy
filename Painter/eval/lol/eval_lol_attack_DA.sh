TASK=lol_enhance

ATTACK_METHOD=PGD
EPSILON=2
STEPS=10

# """ Attack_VA, Attack_AA, Attack_DA, Attack_ADA """

## Attack_DA distribution
EXP=Attack_DA
# for EXP_ID in attack_A attack_B attack_C attack_AB attack_AC attack_BC attack_ABC
for EXP_ID in attack_C attack_AB attack_ABC
do
    # for LAM_AC in 0.1 0.01 0.001
    for LAM_AC in 10 1 0.0001
    do
# 
    OUT_PATH="/hhd3/ld/painter_sammy_output/${TASK}/${EXP}/${ATTACK_METHOD}_eps${EPSILON}_steps${STEPS}/lamAC_${LAM_AC}/${EXP_ID}"

    DST_DIR="${OUT_PATH}/output/"
    SAVE_DATA_PATH="${OUT_PATH}/save_data/"

    CUDA_VISIBLE_DEVICES=1 python painter_inference_lol.py \
        --ckpt_path /hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth \
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
    done
done