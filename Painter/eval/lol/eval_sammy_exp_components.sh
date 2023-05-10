# EXP_ID=baseline
# EXP_ID=POS_A_mask_A
# EXP_ID=POS_B_mask_B
# EXP_ID=POS_AB_mask_AB  # same as format_4_1_a
# EXP_ID=POS_A_random_A_same_task
# EXP_ID=POS_B_random_B_same_task
# EXP_ID=POS_A_random_A_other_task
# EXP_ID=POS_B_random_B_other_task
# EXP_ID=POS_A_animeGAN_A
# TRANSFER_IMG="/hhd3/ld/data/light_enhance/AnimeGANv2/100_low.png"
EXP_ID=POS_B_animeGAN_B
TRANSFER_IMG="/hhd3/ld/data/light_enhance/AnimeGANv2/100_high.png"


# EXP_ID=exp_POS_A_ood_A_animeGAN
# EXP_ID=exp_POS_B_ood_B_animeGAN
# EXP_ID=POS_AB_exchange_AB
# EXP_ID=exp_POS_A_random_A_other_task_coco_gt
# EXP_ID=POS_B_random_B_other_task_Flickr
# EXP_ID=exp_POS_B_random_B_other_task_random


### 15706M


CUDA_VISIBLE_DEVICES=3 python eval/lol/painter_inference_lol_exp_components.py \
    --ckpt_path /hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth \
    --exp_id ${EXP_ID} \
    --transfer_img ${TRANSFER_IMG} 



## bash eval/lol/eval_sammy_exp_components.sh