TASK=lol_enhance

ATTACK_METHOD=PGD
EPSILON=2
STEPS=10

## Attack_AA Alignment attack: break AB --> mask_B
EXP=Attack_ADA  

for EXP_ID in attack_C attack_AB attack_ABC
do
    for LAM_AC in 0.001 # 0.1 0.01
    do
        for LAM_AB in 0.1 0.01 0.001 # 0.01 0.001
        do 
            for MASK_RATIO in 0.75 0.5 0.25 0.1 # 0.75 0.5 0.25 0.1
            do

            OUT_PATH="/hhd3/ld/painter_sammy_output/${TASK}/${EXP}/${ATTACK_METHOD}_eps${EPSILON}_steps${STEPS}/ignore_D_loss/mask_ratio_${MASK_RATIO}/lamAB_${LAM_AB}/lamAC_${LAM_AC}/${EXP_ID}"
            DST_DIR="${OUT_PATH}/output/"
            SAVE_DATA_PATH="${OUT_PATH}/save_data/"

            CUDA_VISIBLE_DEVICES=5 python painter_inference_lol.py \
                --ckpt_path /hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth \
                --task ${TASK} \
                --exp ${EXP} \
                --dst_dir ${DST_DIR} \
                --save_data_path ${SAVE_DATA_PATH} \
                --exp_id ${EXP_ID} \
                --attack_method ${ATTACK_METHOD} \
                --epsilon ${EPSILON} \
                --num_steps ${STEPS} \
                --mask_B \
                --ignore_D_loss \
                --lam_AB ${LAM_AB} \
                --mask_ratio ${MASK_RATIO} \
                --break_AC \
                --lam_AC ${LAM_AC} \
                --with_B \
                --save_adv
            done 
        done 
    done 
done


