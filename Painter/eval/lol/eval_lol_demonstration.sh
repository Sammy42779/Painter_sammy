TASK=lol_enhance


# """ Demonstration """
STYLE_CHANGE_A='/hhd3/ld/data/light_enhance/AnimeGANv2/100_low.png'
STYLE_CHANGE_B='/hhd3/ld/data/light_enhance/AnimeGANv2/100_high.png'
EXP=Demonstration

for EXP_ID in POS_A_mask_A POS_B_mask_B POS_AB_mask_AB POS_A_random_A_same_task POS_B_random_B_same_task POS_A_random_A_other_task POS_B_random_B_other_task POS_A_animeGAN_A POS_B_animeGAN_B POS_AB_animeGAN_AB
do 
OUT_PATH="/hhd3/ld/painter_sammy_output/${TASK}/${EXP}/${EXP_ID}"

DST_DIR="${OUT_PATH}/output/"
SAVE_DATA_PATH="${OUT_PATH}/save_data/"

CUDA_VISIBLE_DEVICES=7 python painter_inference_lol.py \
    --ckpt_path /hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth \
    --task ${TASK} \
    --exp ${EXP} \
    --dst_dir ${DST_DIR} \
    --save_data_path ${SAVE_DATA_PATH} \
    --exp_id ${EXP_ID} \
    --style_change_A ${STYLE_CHANGE_A} \
    --style_change_B ${STYLE_CHANGE_B} \
    --save_demon
done 






