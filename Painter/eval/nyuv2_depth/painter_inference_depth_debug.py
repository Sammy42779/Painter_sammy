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

    """
    %%%%%%%%%%%%%这里传进来的数据就应该是对抗样本了
    """

    x = torch.tensor(img)  # shape [896, 448, 3] 将图片转换为tensor
    # torch.save(x, '/ssd1/ld/ICCV2023/Painter_sammy/rgb.pt')
    x = x.unsqueeze(dim=0)  # shape [1, 896, 448, 3] 在第0维度增加一个维度, 这是因为神经网络通常需要一个批量（batch）的输入，所以在第0维度添加一个维度表示批量大小为1。
    x = torch.einsum('nhwc->nchw', x)  # shape [1, 3, 896, 448] 将nhwc转换为nchw  将输入张量的格式从NHWC（批量大小 x 高度 x 宽度 x 通道数）转换为NCHW（批量大小 x 通道数 x 高度 x 宽度）。

    tgt = torch.tensor(tgt)
    # torch.save(tgt, '/ssd1/ld/ICCV2023/Painter_sammy/depth.pt')
    tgt = tgt.unsqueeze(dim=0)
    tgt = torch.einsum('nhwc->nchw', tgt)

    bool_masked_pos = torch.zeros(model.patch_embed.num_patches)  # [896, 448]的图像, patch_size=16, 则num_patches=(896/16)*(448/16)=1568
    bool_masked_pos[model.patch_embed.num_patches//2:] = 1  # [0, 0, 0, ..., 1, 1, 1, ..., 1, 1, 1] 后半部分为1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)  # [1, 1568], 变长行向量
    valid = torch.ones_like(tgt)  # valid是一个全1的tensor, shape [1, 3, 896, 448] = tgt, 用于计算loss

    # 进Painter模型的forward部分
    loss, y, mask = model(x.float().to(device), tgt.float().to(device), bool_masked_pos.to(device), valid.float().to(device))
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    output = y[0, y.shape[1]//2:, :, :]
    output = torch.clip((output * imagenet_std + imagenet_mean) * 10000, 0, 10000)
    output = F.interpolate(output[None, ...].permute(0, 3, 1, 2), size=[size[1], size[0]], mode='bilinear').permute(0, 2, 3, 1)[0]
    output = output.mean(-1).int()
    output = Image.fromarray(output.numpy())
    output.save("/ssd1/ld/ICCV2023/Painter_sammy/output.jpg", "JPEG")
    output.save(out_path)
    

def get_args_parser():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser('NYU Depth V2', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt',
                        default='/hhd3/ld/checkpoint/ckpt_Painter/painter_vit_large.pth')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1')
    parser.add_argument('--prompt', type=str, help='prompt image in train set',
                        default='study_room_0005b/rgb_00094')
    parser.add_argument('--input_size', type=int, default=448)
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
                           "nyuv2_depth_inference_{}_{}/".format(ckpt_file, args.prompt))
    print(f'----------dst_dir: {dst_dir}----------')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    img_src_dir = "/hhd3/ld/data/nyu_depth_v2/official_splits/test/"
    img_path_list = glob.glob(img_src_dir + "/*/rgb*g")  # test文件中所有文件夹内部的rgb图片, 应该作为所有的query
    img2_path = "/hhd3/ld/data/nyu_depth_v2/sync/{}.jpg".format(args.prompt)  # 选取的prompt的rgb图片
    tgt_path = "/hhd3/ld/data/nyu_depth_v2/sync/{}.png".format(args.prompt.replace('rgb', 'sync_depth'))  # 选取的prompt图片对应的depth图片
    tgt2_path = tgt_path


    """
    %%%%%%%%%%%%%prompt是预先定好的对应的,那么这里进行替换或者攻击
    1. 替换甚至不需要resize, 直接替换, 因为读取后会自动resize然后norm
    2. 只有生成对抗样本的时候要考虑norm的问题
    """



    res, hres = args.input_size, args.input_size  # image的分辨率: 448x448

    for img_path in tqdm.tqdm(img_path_list):
        room_name = img_path.split("/")[-2]  # 各种室内场景, 每一个室内
        img_name = img_path.split("/")[-1].split(".")[0]  # rgb图像名
        out_path = dst_dir + "/" + room_name + "_" + img_name + ".png"  # 猜测是output的深度图, 保存为png格式
        img = Image.open(img_path).convert("RGB") # query image
        size = img.size
        img = img.resize((res, hres))  # [0,255]
        img = np.array(img) / 255.  # [0.0, 1.0]
        img2 = Image.open(img2_path).convert("RGB")
        img2 = img2.resize((res, hres))
        img2 = np.array(img2) / 255.
        img = np.concatenate((img2, img), axis=0)  # axis=0代表竖着拼接, axis=1代表横着拼接, 这里是将prompt的rgb图像和query的rgb图像竖着拼接在一起, 也就是A和C图
        assert img.shape == (2 * res, res, 3)  # [896, 448, 3], assert判断拼接后的图像的shape是否符合预期
        # normalize by ImageNet mean and std: 减去均值, 除以标准差: 图像的像素值将具有零均值和单位方差(但对于其他数据集,不一定零均值), 这有助于提高模型的性能和收敛速度. 应该是将数据从椭圆变成正圆的分布?
        img = img - imagenet_mean  # 0均值, 1标准差
        img = img / imagenet_std

        tgt = Image.open(tgt_path)  # 仅仅是open不转换为RGB 深度图像通常表示距离信息, 其值可能在不同范围内, 例如0到10000  此时的范围是[0,10000]
        tgt = np.array(tgt) / 10000.  # 缩放 max=1, min=0, [0.0, 1.0]
        tgt = tgt * 255  # max=255, min=0 [0.0, 255.0]
        tgt = Image.fromarray(tgt).convert("RGB")  # 转换为Image对象,重新读取图片 [0,255], 这里是把单通道的深度图变成RGB的深度图, 方式是三个通道的值相等    
        tgt = tgt.resize((res, hres))
        tgt = np.array(tgt) / 255.
        tgt2 = Image.open(tgt2_path)
        tgt2 = np.array(tgt2) / 10000.
        tgt2 = tgt2 * 255
        tgt2 = Image.fromarray(tgt2).convert("RGB")
        tgt2 = tgt2.resize((res, hres))
        tgt2 = np.array(tgt2) / 255.
        tgt = np.concatenate((tgt2, tgt), axis=0)

        assert tgt.shape == (2 * res, res, 3)
        # normalize by ImageNet mean and std
        tgt = tgt - imagenet_mean
        tgt = tgt / imagenet_std

        torch.manual_seed(2)
        run_one_image(img, tgt, size, model_painter, out_path, device)  # model_painter是预训练好的模型, out_path是输出的深度图
        # 一张图一张图的处理, 一张图处理完之后, 就会保存到out_path中
