# !/bin/bash

# 只能单卡跑，15706M 3min左右
# 654 test images
## 108服务器: /data1/; 110服务器: /hhd3/


###  把B替换成B1, 再攻击AB和B


set -x

JOB_NAME="painter_vit_large"  # 
CKPT_FILE="painter_vit_large.pth"
PROMPT="study_room_0005b/rgb_00094"


# ATTACK_ID=attack_B
# ATTACK_ID=attack_AB
ATTACK_ID=attack_AC
# ATTACK_ID=attack_BC
# ATTACK_ID=attack_ABC
# ATTACK_ID=none
EPSILON=8
# ALPHA=128

# ATTACK=PGD
ATTACK=FGSM
STEP=1




MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"
CKPT_PATH="/hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth"
DST_DIR="/hhd3/ld/data/nyu_depth_v2/output_attack_changeA/${ATTACK}_${STEP}/${ATTACK_ID}_${EPSILON}"

# inference
CUDA_VISIBLE_DEVICES=4 python eval/nyuv2_depth/painter_inference_depth_attack_with_clip_changeA.py \
  --ckpt_path /hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth \
  --model ${MODEL} \
  --prompt ${PROMPT} \
  --attack_id ${ATTACK_ID} \
  --epsilon ${EPSILON} \
  --attack_method ${ATTACK} \
  --num_steps ${STEP} 


CUDA_VISIBLE_DEVICES=4 python eval/nyuv2_depth/eval_with_pngs.py \
  --pred_path ${DST_DIR} \
  --gt_path /hhd3/ld/data/nyu_depth_v2/official_splits/test/ \
  --dataset nyu --min_depth_eval 1e-3 --max_depth_eval 10 --eigen_crop


########### bash eval/nyuv2_depth/eval_sammy_attack_with_clip_changeA.sh 