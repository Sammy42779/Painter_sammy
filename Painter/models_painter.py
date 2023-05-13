# --------------------------------------------------------
# Images Speak in Images: A Generalist Painter for In-Context Visual Learning (https://arxiv.org/abs/2212.02499)
# Github source: https://github.com/baaivision/Painter
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Xinlong Wang, Wen Wang
# Based on MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------'

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

##########################
import fvcore.nn.weight_init as weight_init
from detectron2.layers import CNNBlockBase, Conv2d, get_norm
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Mlp

from util.vitdet_utils import (
    PatchEmbed,
    add_decomposed_rel_pos,
    get_abs_pos,
    window_partition,
    window_unpartition,
    LayerNorm2D,
)


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class ResBottleneckBlock(CNNBlockBase):
    """
    The standard bottleneck residual block without the last activation layer.
    It contains 3 conv layers with kernels 1x1, 3x3, 1x1.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        norm="LN",
        act_layer=nn.GELU,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            act_layer (callable): activation for all conv layers.
        """
        super().__init__(in_channels, out_channels, 1)

        self.conv1 = Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = get_norm(norm, bottleneck_channels)
        self.act1 = act_layer()

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            3,
            padding=1,
            bias=False,
        )
        self.norm2 = get_norm(norm, bottleneck_channels)
        self.act2 = act_layer()

        self.conv3 = Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = get_norm(norm, out_channels)

        for layer in [self.conv1, self.conv2, self.conv3]:
            weight_init.c2_msra_fill(layer)
        for layer in [self.norm1, self.norm2]:
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()
        # zero init last norm layer.
        self.norm3.weight.data.zero_()
        self.norm3.bias.data.zero_()

    def forward(self, x):
        out = x
        for layer in self.children():
            out = layer(out)

        out = x + out
        return out


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        use_residual_block=False,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_residual_block (bool): If True, use a residual block after the MLP block.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

        self.window_size = window_size

        self.use_residual_block = use_residual_block
        if use_residual_block:
            # Use a residual block with bottleneck channel as dim // 2
            self.residual = ResBottleneckBlock(
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 2,
                norm="LN",
                act_layer=act_layer,
            )

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)  # 通过QKV计算attention
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.use_residual_block:
            x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return x


class Painter(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(
             self,
             img_size=224,
             patch_size=16,
             in_chans=3,
             embed_dim=1024,
             depth=24,
             num_heads=16,
             mlp_ratio=4.,
             qkv_bias=True,
             drop_path_rate=0.,
             norm_layer=nn.LayerNorm,
             act_layer=nn.GELU,
             use_abs_pos=True,
             use_rel_pos=False,
             rel_pos_zero_init=True,
             window_size=0,
             window_block_indexes=(),
             residual_block_indexes=(),
             use_act_checkpoint=False,
             pretrain_img_size=224,
             pretrain_use_cls_token=True,  # 表示在Transformer编码器的输入序列中, 将会添加一个特殊的分类CLS token
             out_feature="last_feat",
             decoder_embed_dim=128,
             loss_func="smoothl1",
             ):
        super().__init__()

        # --------------------------------------------------------------------------
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.patch_embed.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.segment_token_x = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.segment_token_y = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim), requires_grad=True)
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                use_residual_block=i in residual_block_indexes,
                input_size=(img_size[0] // patch_size, img_size[1] // patch_size),
            )
            if use_act_checkpoint:
                block = checkpoint_wrapper(block)
            self.blocks.append(block)

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(embed_dim*4, patch_size ** 2 * self.decoder_embed_dim, bias=True)
        self.decoder_pred = nn.Sequential(
                nn.Conv2d(self.decoder_embed_dim, self.decoder_embed_dim, kernel_size=3, padding=1, ),
                LayerNorm2D(self.decoder_embed_dim),
                nn.GELU(),
                nn.Conv2d(self.decoder_embed_dim, 3, kernel_size=1, bias=True),
        )
        # --------------------------------------------------------------------------
        self.loss_func = loss_func

        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.segment_token_x, std=.02)
        torch.nn.init.normal_(self.segment_token_y, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == 2 * imgs.shape[3] and imgs.shape[2] % p == 0

        w = imgs.shape[3] // p
        h = w * 2
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        w = int((x.shape[1]*0.5)**.5)
        h = w * 2
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    def forward_encoder(self, imgs, tgts, bool_masked_pos):  # img=[1,3,896,448], tgts=[1,3,896,448]
        # embed patches, 初始化的embed_dim=1024, 没有通道信息: 将输入图像分割成大小相等的patches, 然后将这些子块嵌入到一个更低维的表示中, 应该是变成一长条patch序列, 每个patch由n维向量表示
        x = self.patch_embed(imgs)   # [1,56,28,1024] B C H W -> B H W C
        y = self.patch_embed(tgts)   # [1,56,28,1024] B C H W -> B H W C
        batch_size, Hp, Wp, _ = x.size()
        seq_len = Hp * Wp

        mask_token = self.mask_token.expand(batch_size, Hp, Wp, -1)  # [1,1,1,1024] -> [1,56,28,1024] 这里的-1就是初始化时候的embed_dim=1024
        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token).reshape(-1, Hp, Wp, 1)  # 用于在指定维度上增加一个大小为1的维度
        y = y * (1 - w) + mask_token * w  # 这里是构建了输入的target, mask_token就是初始化的要预测的图像, 即对应的任务的预测结果

        # add pos embed w/o cls token  without/   因为后面要将x和y concat 起来，所以要用segment_token_x/y来区分是输入图像还是目标图像, 这个token不加给CLS token
        x = x + self.segment_token_x  # 广播机制, 给x的每个patch都加上segment_token_x, 表明这是输入图像
        y = y + self.segment_token_y
        if self.pos_embed is not None:  # self.pos_embed = [1,197,1024], 其中197=(224/16)*(224/16)+1[CLS token]
            x = x + get_abs_pos(  # 使用 get_abs_pos 函数获取绝对位置嵌入并将其添加到输入和目标图像张量上, CLS token不加位置编码
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )  # 如果num_positions不等于hw的大小, 则需要resize, 这里的get_abs_pos输出shape=[1,56,28,1024]和x一致
            y = y + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (y.shape[1], y.shape[2])
            )
        """
        %%%%%%%%%%%%%在这之上是在构建输入的x和y数据样式
        那么应该在这之后进行对抗样本的构建
        1. 对抗样本不能先norm,因为对抗样本的范围在0-1之间
            1.1 需要先生成数据形式,然后再进行norm用于后面ViT提取特征
        2. mask_token, segment_token, position_embed都不能添加扰动, 只能是原始图像上加扰动
            2.1 那么创建对抗样本的位置需要改变, 不在这里
        """
        merge_idx = 2
        x = torch.cat((x, y), dim=0)  # 按batch_size维度拼接, [2,56,28,1024]
        # apply Transformer blocks
        out = [] 
        for idx, blk in enumerate(self.blocks):  # 这里是用ViT做encoder, 24 blocks for ViT-large
            x = blk(x)  # 输出的shape没有改变 [2,56,28,1024]
            if idx == merge_idx:  # merge the early features of the input image and the output image, 3 blocks, 所以之后变成1的batch size了?
                x = (x[:x.shape[0]//2] + x[x.shape[0]//2:]) * 0.5  # ?
            if idx in [5, 11, 17, 23]:
                out.append(self.norm(x))
        return out  # idx=5,11,17,23时, 保留这几层的输出

    def forward_encoder_ac(self, imgs, tgts, bool_masked_pos):  # img=[1,3,896,448], tgts=[1,3,896,448]
        # embed patches, 初始化的embed_dim=1024, 没有通道信息: 将输入图像分割成大小相等的patches, 然后将这些子块嵌入到一个更低维的表示中, 应该是变成一长条patch序列, 每个patch由n维向量表示
        x = self.patch_embed(imgs)   # [1,56,28,1024] B C H W -> B H W C
        y = self.patch_embed(tgts)   # [1,56,28,1024] B C H W -> B H W C
        batch_size, Hp, Wp, _ = x.size()
        seq_len = Hp * Wp

        mask_token = self.mask_token.expand(batch_size, Hp, Wp, -1)  # [1,1,1,1024] -> [1,56,28,1024] 这里的-1就是初始化时候的embed_dim=1024
        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token).reshape(-1, Hp, Wp, 1)  # 用于在指定维度上增加一个大小为1的维度
        w[:] = 1
        assert torch.sum(1-w) == 0
        
        y = y * (1 - w) + mask_token * w  # 这里是构建了输入的target, mask_token就是初始化的要预测的图像, 即对应的任务的预测结果

        # add pos embed w/o cls token  without/   因为后面要将x和y concat 起来，所以要用segment_token_x/y来区分是输入图像还是目标图像, 这个token不加给CLS token
        x = x + self.segment_token_x  # 广播机制, 给x的每个patch都加上segment_token_x, 表明这是输入图像
        y = y + self.segment_token_y
        if self.pos_embed is not None:  # self.pos_embed = [1,197,1024], 其中197=(224/16)*(224/16)+1[CLS token]
            x = x + get_abs_pos(  # 使用 get_abs_pos 函数获取绝对位置嵌入并将其添加到输入和目标图像张量上, CLS token不加位置编码
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )  # 如果num_positions不等于hw的大小, 则需要resize, 这里的get_abs_pos输出shape=[1,56,28,1024]和x一致
            y = y + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (y.shape[1], y.shape[2])
            )
        """
        %%%%%%%%%%%%%在这之上是在构建输入的x和y数据样式
        那么应该在这之后进行对抗样本的构建
        1. 对抗样本不能先norm,因为对抗样本的范围在0-1之间
            1.1 需要先生成数据形式,然后再进行norm用于后面ViT提取特征
        2. mask_token, segment_token, position_embed都不能添加扰动, 只能是原始图像上加扰动
            2.1 那么创建对抗样本的位置需要改变, 不在这里
        """
        merge_idx = 2
        x = torch.cat((x, y), dim=0)  # 按batch_size维度拼接, [2,56,28,1024]
        x = x.reshape(x.shape[0], 2, Hp//2, Wp, x.shape[-1]) #[col=2, row=2, h=28, w=28, C=1024]
        x = x.flatten(0,1)
        
        # apply Transformer blocks
        out = [] 
        for idx, blk in enumerate(self.blocks):  # 这里是用ViT做encoder, 24 blocks for ViT-large
            x = blk(x)  # 输出的shape没有改变 [4,28,28,1024]
            if idx == merge_idx:  # merge the early features of the input image and the output image, 3 blocks, 所以之后变成1的batch size了?
                x = (x[:x.shape[0]//2] + x[x.shape[0]//2:]) * 0.5  # (row=2, 28, 28, 1024)
            if idx in [5, 11, 17, 23]:
                out.append(self.norm(x).split([1, 1], dim=0))
        out_a = torch.cat([_[0] for _ in out])
        out_c = torch.cat([_[1] for _ in out])
        return out_a, out_c  # idx=5,11,17,23时, 保留这几层的输出

    def forward_decoder(self, x):  
        # predictor projection  解码器的前向传播, 将编码器提取的特征映射回原始图像空间
        x = torch.cat(x, dim=-1)  # 按最后一个维度拼接 4个[1,56,28,1024] -> [1,56,28,1024*4]
        x = self.decoder_embed(x)  # [1,56,28,16384] 全连接层, 它的作用是将编码器（ViT）提取的特征映射到一个适当的尺寸, 以便后续解码器步骤可以将这些特征转换回原始分辨率
        p = self.patch_size  # 16
        h, w = x.shape[1], x.shape[2]  # 56, 28
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.decoder_embed_dim))  # [1,56,28,16,16,64] self.decoder_embed_dim=64
        x = torch.einsum('nhwpqc->nchpwq', x)  # [1,64,56,16,28,16] 交换维度
        x = x.reshape(shape=(x.shape[0], -1, h * p, w * p))  # [1,64,896,448] 64=16*4, 896=56*16, 448=28*16
 
        x = self.decoder_pred(x) # Bx3xHxW  [1,3,896,448] 3是RGB通道
        return x

    def forward_loss(self, pred, tgts, mask, valid):
        """
        tgts: [N, 3, H, W]
        pred: [N, 3, H, W]
        mask: [N, L], 0 is keep, 1 is remove, 
        valid: [N, 3, H, W]
        """
        mask = mask[:, :, None].repeat(1, 1, self.patch_size**2 * 3)
        mask = self.unpatchify(mask)

        # ignore if the unmasked pixels are all zeros
        imagenet_mean=torch.tensor([0.485, 0.456, 0.406]).to(tgts.device)[None, :, None, None]
        imagenet_std=torch.tensor([0.229, 0.224, 0.225]).to(tgts.device)[None, :, None, None]
        inds_ign = ((tgts * imagenet_std + imagenet_mean) * (1 - 1.*mask)).sum((1, 2, 3)) < 100*3
        if inds_ign.sum() > 0:
            valid[inds_ign] = 0.

        mask = mask * valid   ## [1,3,896,448]  mask[:,:,:448,:]均为0, mask[:,:,448:,:]均为1, valid全为1

        target = tgts
        if self.loss_func == "l1l2":
            loss = ((pred - target).abs() + (pred - target) ** 2.) * 0.5
        elif self.loss_func == "l1":
            loss = (pred - target).abs()
        elif self.loss_func == "l2":
            loss = (pred - target) ** 2.
        elif self.loss_func == "smoothl1":
            loss = F.smooth_l1_loss(pred, target, reduction="none", beta=0.01)
        loss = (loss * mask).sum() / (mask.sum() + 1e-2)  # mean loss on removed patches  只考虑D图的loss
        return loss

    def forward(self, imgs, tgts, bool_masked_pos=None, valid=None):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((imgs.shape[0], self.patch_embed.num_patches), dtype=torch.bool).to(imgs.device)
        else:
            bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)   # 变成True/False
        latent = self.forward_encoder(imgs, tgts, bool_masked_pos)  ## 得到encoder后的输出, 4个层的输出, 每一个输出的shape是[1,56,28,1024]
        pred = self.forward_decoder(latent)  # [N, L, p*p*3]
        loss = self.forward_loss(pred, tgts, bool_masked_pos, valid)
        return loss, self.patchify(pred), bool_masked_pos



def painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1(**kwargs):
    model = Painter(
        img_size=(896, 448), patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        drop_path_rate=0.1, window_size=14, qkv_bias=True,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=(list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + list(range(9, 11)) + \
                                list(range(12, 14)), list(range(15, 17)), list(range(18, 20)), list(range(21, 23))),
        residual_block_indexes=[], use_rel_pos=True, out_feature="last_feat",
        decoder_embed_dim=64,
        loss_func="smoothl1",
        **kwargs)
    return model


def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)

