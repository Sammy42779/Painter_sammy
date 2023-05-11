



# ATTACK_ID=attack_A
# ATTACK_ID=attack_B
ATTACK_ID=attack_C
# ATTACK_ID=attack_AB
# ATTACK_ID=attack_AC
# ATTACK_ID=attack_BC
# ATTACK_ID=attack_ABC
# ATTACK_ID=none
EPSILON=8
# ALPHA=128

ATTACK=PGD
# ATTACK=FGSM
STEP=10


#### PGD10步 2mins



CUDA_VISIBLE_DEVICES=4 python eval/lol/painter_inference_lol_attack_with_clip.py \
    --ckpt_path /hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth \
    --attack_id ${ATTACK_ID} \
    --epsilon ${EPSILON} \
    --attack_method ${ATTACK} \
    --num_steps ${STEP} 



## bash eval/lol/eval_sammy_attack_with_clip.sh