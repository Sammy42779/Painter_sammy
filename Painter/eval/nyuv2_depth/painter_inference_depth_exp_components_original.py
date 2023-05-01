# --------------------------------------------------------
# Images Speak in Images: A Generalist Painter for In-Context Visual Learning (https://arxiv.org/abs/2212.02499)
# Github source: https://github.com/baaivision/Painter
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Xinlong Wang, Wen Wang
# Based on MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------'

import sys
import os
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import glob
import tqdm

import matplotlib.pyplot as plt
from PIL import Image

import random

## change sys path_dir based on the server 修改当前Painter路径，不是数据集路径
## 108: '/ssd1/ld/ICCV2023/Painter_sammy/Painter'
## 110: '/ssd3/ld/sammy2023/Painter_sammy/Painter'
# sys.path.append('/ssd1/ld/ICCV2023/Painter_sammy/Painter')
sys.path.append('/ssd1/ld/ICCV2023/Painter_sammy/Painter')
import models_painter


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def prepare_model(chkpt_dir, arch='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1'):
    # build model
    model = getattr(models_painter, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cuda:0')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    model.eval()
    return model


def run_one_image(img, tgt, size, model, out_path, device):
    x = torch.tensor(img)
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    tgt = torch.tensor(tgt)
    tgt = tgt.unsqueeze(dim=0)
    tgt = torch.einsum('nhwc->nchw', tgt)

    bool_masked_pos = torch.zeros(model.patch_embed.num_patches)
    bool_masked_pos[model.patch_embed.num_patches//2:] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)
    valid = torch.ones_like(tgt)
    
    loss, y, mask = model(x.float().to(device), tgt.float().to(device), bool_masked_pos.to(device), valid.float().to(device))
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    output = y[0, y.shape[1]//2:, :, :]
    output = torch.clip((output * imagenet_std + imagenet_mean) * 10000, 0, 10000)
    output = F.interpolate(output[None, ...].permute(0, 3, 1, 2), size=[size[1], size[0]], mode='bilinear').permute(0, 2, 3, 1)[0]
    output = output.mean(-1).int()
    output = Image.fromarray(output.numpy())
    output.save(out_path)
    

def get_args_parser():
    parser = argparse.ArgumentParser('NYU Depth V2', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt',
                        default='/hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1')
    parser.add_argument('--prompt', type=str, help='prompt image in train set',
                        default='study_room_0005b/rgb_00094')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--exp_id', type=str, default='exp_mapping_1_1_a')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args_parser()

    ckpt_path = args.ckpt_path

    path_splits = ckpt_path.split('/')
    ckpt_dir, ckpt_file = path_splits[-2], path_splits[-1]

    model_painter = prepare_model(ckpt_path, args.model)
    print('Model loaded.')

    device = torch.device("cuda")
    model_painter.to(device)

    ## ## change based on the server
    ## 108: /data1/; 110: /hhd3/
    dst_dir = os.path.join('/hhd3/ld/data/nyu_depth_v2/'
                           "nyuv2_depth_inference_{}_{}_{}/".format(ckpt_file, args.prompt, args.exp_id))
    print(f'----------dst_dir: {dst_dir}----------')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    img_src_dir = "/hhd3/ld/data/nyu_depth_v2/official_splits/test/"
    img_path_list = glob.glob(img_src_dir + "/*/rgb*g")

    flickr_list = glob.glob('/hhd3/ld/data/flickr30k-images/*.jpg')
    
    if args.exp_id == 'exp_mapping_1_1_a':
        ## 1-1(a): 将图B替换为COCO Panoptic Segmentation最佳prompt /ssd1/ld/ICCV2023/Painter_sammy/Painter/eval/coco_panoptic/eval.sh
        ## 替换tgt_path为COCO Panoptic Segmentation最佳prompt  不再是深度图了
    
        img2_path = "/hhd3/ld/data/nyu_depth_v2/sync/{}.jpg".format(args.prompt)
        
        tgt_path = "/hhd3/ld/data/COCO2017/annotations/panoptic_train2017/000000391460.png"
        # tgt_path = "/hhd3/ld/data/nyu_depth_v2/sync/{}.png".format(args.prompt.replace('rgb', 'sync_depth'))
        tgt2_path = tgt_path

    elif args.exp_id == 'exp_mapping_1_1_b':
        ## 1-1(b): 将图B替换为Depth任务这个prompt的其他depth-gt图  
        ## 替换tgt_path为随机的depth-gt图   仍然是深度图
        img2_path = "/hhd3/ld/data/nyu_depth_v2/sync/{}.jpg".format(args.prompt)
       
        tgt_path = "/hhd3/ld/data/nyu_depth_v2/official_splits/train/bathroom/sync_depth_00699.png" 
        # "/hhd3/ld/data/nyu_depth_v2/official_splits/train/excercise_room/sync_depth_00348.png"
        tgt2_path = tgt_path
    
    elif args.exp_id == 'exp_input_distribution_2_1_a':
        ## 2-1(a): 将图A换成train OOD图像: 替换input distribution
        ## 替换img2_path为Flickr30K里的图像

        # img2_path = "/hhd3/ld/data/nyu_depth_v2/sync/{}.jpg".format(args.prompt)
        # flickr_img_dir = '/hhd3/ld/data/flickr30k-images/2688337844.jpg'
        flickr_img_dir = random.choice(flickr_list)
        img2_path = flickr_img_dir

        tgt_path = "/hhd3/ld/data/nyu_depth_v2/sync/{}.png".format(args.prompt.replace('rgb', 'sync_depth'))
        tgt2_path = tgt_path
    
    elif args.exp_id == 'exp_output_distribution_3_1_a':
        ## 3-1(b): 将图B替换为train OOD图像: 替换output distribution
        img2_path = "/hhd3/ld/data/nyu_depth_v2/sync/{}.jpg".format(args.prompt)
        
        flickr_img_dir = '/hhd3/ld/data/flickr30k-images/2688337844.jpg'
        tgt_path = flickr_img_dir
        # tgt_path = "/hhd3/ld/data/nyu_depth_v2/sync/{}.png".format(args.prompt.replace('rgb', 'sync_depth'))
        tgt2_path = tgt_path

    
    else:  # baseline
        img2_path = "/hhd3/ld/data/nyu_depth_v2/sync/{}.jpg".format(args.prompt)  # 选取的prompt的rgb图片
        tgt_path = "/hhd3/ld/data/nyu_depth_v2/sync/{}.png".format(args.prompt.replace('rgb', 'sync_depth'))  # 选取的prompt图片对应的depth图片
        tgt2_path = tgt_path

    res, hres = args.input_size, args.input_size

    i = 0
    blank_image = np.zeros((res, hres, 3), dtype=np.float64)

    for img_path in tqdm.tqdm(img_path_list):
        room_name = img_path.split("/")[-2]
        img_name = img_path.split("/")[-1].split(".")[0]
        out_path = dst_dir + "/" + room_name + "_" + img_name + ".png"
        img = Image.open(img_path).convert("RGB")
        size = img.size
        img = img.resize((res, hres))
        img = np.array(img) / 255.
        if args.exp_id == 'exp_format_4_1_a':
            # 没有format, 没有prompt, 因此img2和tgt2都是空白图像
            img2 = blank_image
        else:
            img2 = Image.open(img2_path).convert("RGB")
            img2 = img2.resize((res, hres))
            img2 = np.array(img2) / 255.
        img = np.concatenate((img2, img), axis=0)
        assert img.shape == (2 * res, res, 3)
        # normalize by ImageNet mean and std
        img = img - imagenet_mean
        img = img / imagenet_std


        """
        对于更换了的prompt图像的处理有变化
        如果不是深度图, 不需要除以10000, 如果不确定要除以多少, 就先看看图像的min max值, 约束到0-1之间就可以, 然后再乘以255
        """
        if args.exp_id in ['exp_mapping_1_1_a', 'exp_output_distribution_3_1_a']:
            tgt = Image.open(tgt_path)
            tgt = tgt.resize((res, hres))
            tgt = np.array(tgt) / 255.
            tgt2 = Image.open(tgt2_path)
            tgt2 = tgt2.resize((res, hres))
            tgt2 = np.array(tgt2) / 255.
            tgt = np.concatenate((tgt2, tgt), axis=0)
        elif args.exp_id in ['exp_baseline', 'exp_mapping_1_1_b', 'exp_input_distribution_2_1_a']:
            tgt = Image.open(tgt_path)
            tgt = np.array(tgt) / 10000.
            tgt = tgt * 255
            tgt = Image.fromarray(tgt).convert("RGB")
            tgt = tgt.resize((res, hres))
            tgt = np.array(tgt) / 255.
            tgt2 = Image.open(tgt2_path)
            tgt2 = np.array(tgt2) / 10000.
            tgt2 = tgt2 * 255
            tgt2 = Image.fromarray(tgt2).convert("RGB")
            tgt2 = tgt2.resize((res, hres))
            tgt2 = np.array(tgt2) / 255.
            tgt = np.concatenate((tgt2, tgt), axis=0)
        elif args.exp_id == 'exp_format_4_1_a':
            # 没有format, 没有prompt, 因此img2和tgt2都是空白图像
            tgt = blank_image
            tgt2 = blank_image
            tgt = np.concatenate((tgt2, tgt), axis=0)
        

        assert tgt.shape == (2 * res, res, 3)
        # normalize by ImageNet mean and std
        tgt = tgt - imagenet_mean
        tgt = tgt / imagenet_std


        if i <= 2:
            np.save(f'/ssd1/ld/ICCV2023/Painter_sammy/img2_prompt_{args.exp_id}_{i}.npy', img)
            np.save(f'/ssd1/ld/ICCV2023/Painter_sammy/tgt2_prompt_{args.exp_id}_{i}.npy', tgt)
        i += 1
        torch.manual_seed(2)
        run_one_image(img, tgt, size, model_painter, out_path, device)
