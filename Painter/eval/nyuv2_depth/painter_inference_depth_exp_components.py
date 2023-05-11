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

sys.path.append('/ssd1/ld/ICCV2023/Painter_sammy/Painter/eval')
from exp_components_utils import *


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
    
    parser.add_argument('--exp_id', type=str, default='baseline')
    parser.add_argument('--transfer_img_A', type=str, default='animeGAN')
    parser.add_argument('--transfer_img_B', type=str, default='animeGAN')

    parser.add_argument('--dst_dir', type=str, default='dst_dir')
    parser.add_argument('--save_data_path', type=str, default='save_data_path')

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
    # dst_dir = os.path.join('/hhd3/ld/data/nyu_depth_v2/component_analysis/'
    #                        "nyuv2_depth_inference_{}_{}_{}/".format(ckpt_file, args.prompt, args.exp_id))
    dst_dir = args.dst_dir
    print(f'----------dst_dir: {dst_dir}----------')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    img_src_dir = "/hhd3/ld/data/nyu_depth_v2/official_splits/test/"
    img_path_list = glob.glob(img_src_dir + "/*/rgb*g")

    img2_path = "/hhd3/ld/data/nyu_depth_v2/sync/{}.jpg".format(args.prompt)
    tgt_path = "/hhd3/ld/data/nyu_depth_v2/sync/{}.png".format(args.prompt.replace('rgb', 'sync_depth'))
    tgt2_path = tgt_path

    """
    修改prompt, 即 A B 图
    """
    input_size = args.input_size
    img2, tgt2 = get_prompt_gt(img2_path, tgt2_path, input_size, args.exp_id, args.transfer_img_A, args.transfer_img_B, task='nyu_depth')

    res, hres = args.input_size, args.input_size

    i = 0
    SEED = random.choice(np.arange(len(img_path_list)))

    # save_data_path = f'/hhd3/ld/data/Painter_root/nyu_depth/component_analysis/'
    save_data_path = args.save_data_path
    os.makedirs(save_data_path, exist_ok=True)

    for img_path in tqdm.tqdm(img_path_list):
        room_name = img_path.split("/")[-2]
        img_name = img_path.split("/")[-1].split(".")[0]
        out_path = dst_dir + "/" + room_name + "_" + img_name + ".png"
        img = Image.open(img_path).convert("RGB")
        size = img.size
        img = img.resize((res, hres))
        img = np.array(img) / 255.
        img = np.concatenate((img2, img), axis=0)
        assert img.shape == (2 * res, res, 3)
        # normalize by ImageNet mean and std
        img = img - imagenet_mean
        img = img / imagenet_std


        """
        对于更换了的prompt图像的处理有变化
        如果不是深度图, 不需要除以10000, 如果不确定要除以多少, 就先看看图像的min max值, 约束到0-1之间就可以, 然后再乘以255
        """
        ## 这部分处理在utils里面已经做过了
        tgt = tgt2  # tgt is not available
        tgt = np.concatenate((tgt2, tgt), axis=0)

        assert tgt.shape == (2 * res, res, 3)
        # normalize by ImageNet mean and std
        tgt = tgt - imagenet_mean
        tgt = tgt / imagenet_std


        if i == SEED:
            np.save(f'{save_data_path}/img2_prompt_{args.exp_id}.npy', img)
            np.save(f'{save_data_path}/tgt2_prompt_{args.exp_id}.npy', tgt)
        i += 1
        torch.manual_seed(2)
        run_one_image(img, tgt, size, model_painter, out_path, device)
