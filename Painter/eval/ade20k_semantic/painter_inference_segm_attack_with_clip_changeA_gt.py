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
import warnings

import requests
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import glob
import tqdm
import random

import matplotlib.pyplot as plt
from PIL import Image
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

sys.path.append('/ssd1/ld/ICCV2023/Painter_sammy/Painter')
import models_painter
from util.ddp_utils import DatasetTest
from util import ddp_utils

sys.path.append('/ssd1/ld/ICCV2023/Painter_sammy/Painter/eval')
from attack_utils_with_clip_basic import *
from constant_utils import *
from exp_components_utils import *


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def get_args_parser():
    parser = argparse.ArgumentParser('ADE20k semantic segmentation', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt', default='/hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1')
    parser.add_argument('--prompt', type=str, help='prompt image in train set',
                        default='ADE_train_00014165')
    parser.add_argument('--input_size', type=int, default=448)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument('--epsilon', default=8, type=int,
                    help='max perturbation (default: 8), need to divide by 255')
    parser.add_argument('--attack_id', type=str, default='attack_AC')
    parser.add_argument('--attack_method', type=str, default='PGD')
    parser.add_argument('--num_steps', default=5, type=int)
    
    parser.add_argument('--dst_dir', type=str, default='dst_dir')
    parser.add_argument('--save_data_path', type=str, default='save_data_path')


    return parser.parse_args()


def prepare_model(chkpt_dir, arch='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1', args=None):
    # build model
    model = getattr(models_painter, arch)()
    model.to("cuda")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def run_one_image(img, tgt, size, model, out_path, device):
    x = torch.tensor(img)
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    tgt = torch.tensor(tgt)
    tgt = tgt.unsqueeze(dim=0)
    tgt = torch.einsum('nhwc->nchw', tgt)

    patch_size = model.module.patch_size
    _, _, h, w = tgt.shape
    num_patches = h * w // patch_size ** 2
    bool_masked_pos = torch.zeros(num_patches)
    bool_masked_pos[num_patches//2:] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)

    valid = torch.ones_like(tgt)
    ### 进模型之前normalize 同时normalize img and tgt
    # loss, y, mask = model(x.float().to(device), tgt.float().to(device), bool_masked_pos.to(device), valid.float().to(device))
    loss, y, mask = model(images_normalize(x).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.float().to(device))
    y = model.module.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    output = y[0, y.shape[1]//2:, :, :]
    output = torch.clip((output * imagenet_std + imagenet_mean) * 255, 0, 255)
    output = F.interpolate(output[None, ...].permute(0, 3, 1, 2), size=[size[1], size[0]], mode='bilinear').permute(0, 2, 3, 1)[0]
    output = output.int()
    output = Image.fromarray(output.numpy().astype(np.uint8))
    output.save(out_path)


if __name__ == '__main__':

    dataset_dir = "/hhd3/ld/data/"
    args = get_args_parser()
    args = ddp_utils.init_distributed_mode(args)
    device = torch.device("cuda")

    ckpt_path = args.ckpt_path
    model = args.model
    prompt = args.prompt
    input_size = args.input_size

    path_splits = ckpt_path.split('/')
    ckpt_dir, ckpt_file = path_splits[-2], path_splits[-1]
    # dst_dir = os.path.join('/hhd3/ld/data/ade20k/output_attack/'
    #                        "ade20k_seg_inference_{}_{}_{}/".format(ckpt_file, args.prompt, args.exp_id))
    # dst_dir = os.path.join(f'/hhd3/ld/data/ade20k/output_attack_change_A/{args.attack_method}_{args.num_steps}/'
    #                        "{}_{}".format(args.attack_id, args.epsilon))
    dst_dir = args.dst_dir
    print(f'----------dst_dir: {dst_dir}----------')

    if ddp_utils.get_rank() == 0:
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        print("output_dir: {}".format(dst_dir))

    model_painter = prepare_model(ckpt_path, model, args=args)
    print('Model loaded.')

    device = torch.device("cuda")
    model_painter.to(device)

    img_src_dir = dataset_dir + "ade20k/images/validation"
    # img_path_list = glob.glob(os.path.join(img_src_dir, "*.jpg"))
    dataset_val = DatasetTest(img_src_dir, input_size, ext_list=('*.jpg',))
    sampler_val = DistributedSampler(dataset_val, shuffle=False)
    data_loader_val = DataLoader(dataset_val, batch_size=1, sampler=sampler_val,
                                 drop_last=False, collate_fn=ddp_utils.collate_fn, num_workers=2)

    ## A图
    # img2_path = dataset_dir + "ade20k/images/training/{}.jpg".format(prompt)
    img2_path = random.choice(COCO_LIST_B)
    tgt2_path = dataset_dir + "ade20k/annotations_with_color/training/{}.png".format(prompt)

    # load the shared prompt image pair  
    img2 = Image.open(img2_path).convert("RGB")
    img2 = img2.resize((input_size, input_size))
    img2 = np.array(img2) / 255.

    ## segment 任务 tgt2 不能convert("RGB")
    tgt2 = Image.open(tgt2_path)
    tgt2 = tgt2.resize((input_size, input_size))
    tgt2 = np.array(tgt2) / 255.

    i = 0
    SEED = random.choice(np.arange(len(data_loader_val)))

    # save_data_path = f'/hhd3/ld/data/Painter_root/ade20k_semantic/change_A/{args.attack_id}_{args.epsilon}/'
    save_data_path = args.save_data_path
    os.makedirs(save_data_path, exist_ok=True)

    model_painter.eval()
    for data in tqdm.tqdm(data_loader_val):
        """ Load an image """
        assert len(data) == 1
        img, img_path, size = data[0]  

        ### 此时的数据都是0-1范围内的数据

        img_name = os.path.basename(img_path)
        out_path = os.path.join(dst_dir, img_name.replace('.jpg', '.png'))

        img = np.concatenate((img2, img), axis=0)
        assert img.shape == (input_size * 2, input_size, 3)
        # normalize by ImageNet mean and std
        # img = img - imagenet_mean
        # img = img / imagenet_std

        # tgt = tgt2  # tgt is not available
        # tgt = gt_path
        gt_name = img_name.replace('.jpg', '.png')
        gt_path = f'/hhd3/ld/data/ade20k/annotations_with_color/validation/{gt_name}' 
        tgt = Image.open(gt_path)
        tgt = tgt.resize((input_size, input_size))
        tgt = np.array(tgt) / 255.
        
        tgt = np.concatenate((tgt2, tgt), axis=0)

        assert tgt.shape == (input_size * 2, input_size, 3)
        # normalize by ImageNet mean and std
        # tgt = tgt - imagenet_mean
        # tgt = tgt / imagenet_std

        # make random mask reproducible (comment out to make it change)
        torch.manual_seed(2)

        adv_img, adv_tgt = get_adv_img_adv_tgt(img, tgt, model_painter, device, args.attack_id, args.attack_method, epsilon=args.epsilon, num_steps=args.num_steps)

        if i == SEED:
            np.save(f'{save_data_path}/img2_prompt.npy', img)
            np.save(f'{save_data_path}/img2_prompt_adv.npy', adv_img)
            np.save(f'{save_data_path}/tgt2_prompt.npy', tgt)
            np.save(f'{save_data_path}/tgt2_prompt_adv.npy', adv_tgt)
        i += 1

        ### 这里出来的对抗样本的范围还是0-1之间, 然后在进入模型之前再做normalize

        run_one_image(adv_img, adv_tgt, size, model_painter, out_path, device)
