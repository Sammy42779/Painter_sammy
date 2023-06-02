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

sys.path.append('/ssd1/ld/ICCV2023/Painter_sammy/Painter')
import models_painter

sys.path.append('/ssd1/ld/ICCV2023/Painter_sammy/Painter/eval')
from constant_utils import *
from demonstration_utils import *

import ADA_utils


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
    
    loss, y, mask = model(images_normalize(x).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.float().to(device))
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

    parser.add_argument('--task', type=str, default='lol_enhance')
    parser.add_argument('--exp', type=str, default='Attack_VA', help='Baseline, Demonstration, Attack_VA(vanilla), Attack_AA(alignment), Attack_DA(distribution), Attack_ADA')
    parser.add_argument('--dst_dir', type=str, default='/ssd1/ld/ICCV2023/Painter_sammy/debug/dst_dir')
    parser.add_argument('--save_data_path', type=str, default='/ssd1/ld/ICCV2023/Painter_sammy/debug/save_data_path')
    parser.add_argument('--exp_id', type=str, default='attack_A', help='Demonstration: POS_exp, Attack: attack_POS')

    parser.add_argument('--style_change_A', type=str, default='animeGAN')
    parser.add_argument('--style_change_B', type=str, default='animeGAN')
    parser.add_argument('--save_demon', action='store_true', help='save demonstration data',
                        default=False)

    parser.add_argument('--random_A', action='store_true', help='random A (OOD) content and then attack vp',
                        default=False)
    parser.add_argument('--random_B', action='store_true', help='random B (ID) content and then attack vp',
                        default=False)
    parser.add_argument('--attack_method', type=str, default='PGD')
    parser.add_argument('--epsilon', default=8, type=int,
                        help='max perturbation (default: 8), need to divide by 255')
    parser.add_argument('--num_steps', default=10, type=int)

    parser.add_argument('--break_AC', action='store_true', help='Distribution Attack',
                        default=False)
    parser.add_argument('--lam_AC', default=0.01, type=float)
    parser.add_argument('--with_B', action='store_true', help='use B to get feat_a and feat_c',
                        default=False)

    parser.add_argument('--mask_B', action='store_true', help='Alignment Attack',
                        default=False)
    parser.add_argument('--ignore_D_loss', action='store_true', help='ignore D loss',
                        default=False)
    parser.add_argument('--lam_AB', default=0.1, type=float)
    parser.add_argument('--mask_ratio', default=0.75, type=float)

    parser.add_argument('--save_adv', action='store_true', help='save adversarial data',
                        default=False)

    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args_parser()

    args_dict = vars(args)

    # make random mask reproducible (comment out to make it change)
    set_seed(args.seed)

    ckpt_path = args.ckpt_path

    path_splits = ckpt_path.split('/')
    ckpt_dir, ckpt_file = path_splits[-2], path_splits[-1]

    model_painter = prepare_model(ckpt_path, args.model)
    print('Model loaded.')

    device = torch.device("cuda")
    model_painter.to(device)

    # dst_dir = os.path.join('models_inference', ckpt_dir,
    #                        "nyuv2_depth_inference_{}_{}/".format(ckpt_file, args.prompt))
    dst_dir = args.dst_dir
    print(f'----------dst_dir: {dst_dir}----------')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    with open(os.path.join(args.dst_dir, 'args.json'), 'w') as f:
        json.dump(args_dict, f)

    img_src_dir = "/hhd3/ld/data/nyu_depth_v2/official_splits/test/"
    img_path_list = glob.glob(img_src_dir + "/*/rgb*g")
    img2_path = "/hhd3/ld/data/nyu_depth_v2/sync/{}.jpg".format(args.prompt)
    tgt_path = "/hhd3/ld/data/nyu_depth_v2/sync/{}.png".format(args.prompt.replace('rgb', 'sync_depth'))
    tgt2_path = tgt_path
    if args.random_A:
        img2_path = random.choice(COCO_LIST_B)
    if args.random_B:
        tgt2_path = random.choice(NYU_LIST_B)

    res, hres = args.input_size, args.input_size

    if args.exp == 'Demonstration':
        img2, tgt2 = get_prompt_gt(img2_path, tgt2_path, args.input_size, 
                                   args.exp_id, args.style_change_A, args.style_change_B, task=args.task)
    else: 
        img2 = Image.open(img2_path).convert("RGB")
        img2 = img2.resize((res, hres))
        img2 = np.array(img2) / 255.
        tgt2 = Image.open(tgt2_path)
        tgt2 = np.array(tgt2) / 10000.
        tgt2 = tgt2 * 255
        tgt2 = Image.fromarray(tgt2).convert("RGB")
        tgt2 = tgt2.resize((res, hres))
        tgt2 = np.array(tgt2) / 255.


    if args.save_adv or args.save_demon:
        i = 0
        SEED = random.choice(np.arange(len(img_path_list)))

        save_data_path = args.save_data_path
        os.makedirs(save_data_path, exist_ok=True)


    for img_path in tqdm.tqdm(img_path_list):
        room_name = img_path.split("/")[-2]
        img_name = img_path.split("/")[-1].split(".")[0]
        out_path = dst_dir + "/" + room_name + "_" + img_name + ".png"
        gt_path = img_src_dir + room_name + "/" + img_name.replace('rgb', 'sync_depth') + ".png"  

        img = Image.open(img_path).convert("RGB")
        size = img.size
        img = img.resize((res, hres))
        img = np.array(img) / 255.
        img = np.concatenate((img2, img), axis=0)
        assert img.shape == (2 * res, res, 3)
        # normalize by ImageNet mean and std
        # img = img - imagenet_mean
        # img = img / imagenet_std

        tgt = Image.open(gt_path)  # Ground Truth
        tgt = np.array(tgt) / 10000.
        tgt = tgt * 255
        tgt = Image.fromarray(tgt).convert("RGB")
        tgt = tgt.resize((res, hres))
        tgt = np.array(tgt) / 255.
        tgt = np.concatenate((tgt2, tgt), axis=0)

        assert tgt.shape == (2 * res, res, 3)
        # normalize by ImageNet mean and std
        # tgt = tgt - imagenet_mean
        # tgt = tgt / imagenet_std

        # torch.manual_seed(2)
        if 'Attack' in args.exp:
            adv_img, adv_tgt = ADA_utils.construct_adv_PGD(img, tgt, model_painter, device, 
                                        epsilon=args.epsilon/255., num_steps=args.num_steps, step_size=2/255.,
                                        exp_method=args.exp, exp_pos=args.exp_id,
                                        break_AC=args.break_AC, lam_AC=args.lam_AC, with_B=args.with_B,
                                        mask_B=args.mask_B, ignore_D_loss=args.ignore_D_loss, lam_AB=args.lam_AB, mask_ratio=args.mask_ratio)
        if args.save_demon or args.save_adv:
            if i == SEED:
                if args.save_demon:
                    np.save(f'{save_data_path}/img2_demon.npy', img)
                    np.save(f'{save_data_path}/tgt2_demon.npy', tgt)
                elif args.save_adv:
                    np.save(f'{save_data_path}/img2_prompt.npy', img)
                    np.save(f'{save_data_path}/img2_prompt_adv.npy', adv_img)
                    np.save(f'{save_data_path}/tgt2_prompt.npy', tgt)
                    np.save(f'{save_data_path}/tgt2_prompt_adv.npy', adv_tgt)
            i += 1

        if 'Attack' in args.exp:
            run_one_image(adv_img, adv_tgt, size, model_painter, out_path, device)
        else:
            run_one_image(img, tgt, size, model_painter, out_path, device)
