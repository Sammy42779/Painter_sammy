# EXP_ID=baseline
# EXP_ID=POS_A_mask_A
# EXP_ID=POS_B_mask_B
# EXP_ID=POS_AB_mask_AB

# EXP_ID=POS_A_random_A_same_task
# EXP_ID=POS_A_random_A_other_task
# EXP_ID=POS_B_random_B_same_task
# EXP_ID=POS_B_random_B_other_task

# EXP_ID=POS_A_animeGAN_A
# EXP_ID=POS_B_animeGAN_B
# EXP_ID=POS_AB_animeGAN_AB


for EXP_ID in POS_A_mask_A POS_B_mask_B POS_AB_mask_AB POS_A_random_A_same_task POS_A_random_A_other_task POS_B_random_B_same_task POS_B_random_B_other_task POS_A_animeGAN_A POS_B_animeGAN_B POS_AB_animeGAN_AB

do
    TRANSFER_IMG_A="/hhd3/ld/data/light_enhance/AnimeGANv2/100_low.png"
    TRANSFER_IMG_B="/hhd3/ld/data/light_enhance/AnimeGANv2/100_high.png"

    ### 15706M
    TASK=lol_enhance
    EXP=component_analysis
    DST_DIR="/hhd3/ld/painter_output/${TASK}/${EXP}/${EXP_ID}/output/"
    SAVE_DATA_PATH="/hhd3/ld/painter_output/${TASK}/${EXP}/${EXP_ID}/save_data/"



    CUDA_VISIBLE_DEVICES=0 python eval/lol/painter_inference_lol_exp_components.py \
        --ckpt_path /hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth \
        --exp_id ${EXP_ID} \
        --transfer_img_A ${TRANSFER_IMG_A} \
        --transfer_img_B ${TRANSFER_IMG_B} \
        --dst_dir ${DST_DIR} \
        --save_data_path ${SAVE_DATA_PATH}



    ## bash eval/lol/eval_sammy_exp_components.sh

done