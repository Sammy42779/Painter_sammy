# !/bin/bash

# 单卡31228M 并行跑也是这么多显存, 但是运行时间会变少
# 5000 val images

set -x

NUM_GPUS=1
JOB_NAME="painter_vit_large"
CKPT_FILE="painter_vit_large.pth"
PROMPT=000000391460

SIZE=560
DIST_THR=19

CKPT_PATH="/hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth"
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"

WORK_DIR="/hhd3/ld/data/COCO2017/eval"


# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}


# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}














# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}



# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













  # inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}














# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













  # inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}














# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}



# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}



# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}












  # inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}














# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}












  # inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}














# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}












  # inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}














# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}













# inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg_cuda.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}










