import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import random
import glob
import json


FLICKR_LIST = glob.glob('/hhd3/ld/data/flickr30k-images/*.jpg')


NYU_LIST_A = glob.glob('/hhd3/ld/data/nyu_depth_v2/official_splits/test/*/rgb*.jpg')
NYU_LIST_B = glob.glob('/hhd3/ld/data/nyu_depth_v2/official_splits/test/*/sync_depth*.png')
NYU_LIST_B_sync = glob.glob('/hhd3/ld/data/nyu_depth_v2/sync/*/sync_depth*.png')   # train时NYU的gt
ADE20K_LIST_A = glob.glob('/hhd3/ld/data/ade20k/images/validation/*.jpg')
ADE20K_LIST_B = glob.glob('/hhd3/ld/data/ade20k/annotations_with_color/validation/*.png')
COCO_LIST_A = glob.glob('/hhd3/ld/data/COCO2017/val2017/*.jpg')
COCO_LIST_B = glob.glob('/hhd3/ld/data/COCO2017/pano_sem_seg/panoptic_segm_train2017_with_color/*.png')
LOL_LIST_A = glob.glob('/hhd3/ld/data/light_enhance/our485/low/*.png')
LOL_LIST_B = glob.glob('/hhd3/ld/data/light_enhance/our485/high/*.png')



imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


imagenet_mean_ts=torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
imagenet_std_ts=torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]

images_normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


criterion = torch.nn.KLDivLoss(reduction='batchmean')

L2 = torch.nn.MSELoss()


mean_pool = nn.AvgPool2d(2)


def reshape(input):
    out = torch.tensor(input)
    out = out.unsqueeze(dim=0)
    out = torch.einsum('nhwc->nchw', out)

    return out


def get_random_mask_B(model, bool_masked_pos, mask_ratio=0.75): # wt
    patch_size = model.patch_size
    input_size = (448, 448)
    # print("Patch size = %s" % str(patch_size))
    window_size = (input_size[0] // patch_size, input_size[1] // patch_size)
    from util.masking_generator import MaskingGenerator
    masked_position_generator = MaskingGenerator(
        window_size, num_masking_patches= int(784 * mask_ratio),
        max_num_patches=None,
        min_num_patches=16,
    )
    mask = masked_position_generator()
    bool_masked_pos[:, :bool_masked_pos.shape[1]//2] = torch.from_numpy(mask.reshape(1, -1))
    return bool_masked_pos


def get_masked_pos(model , mask_B=False, mask_ratio=0.75):

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    else:
        model = model

    bool_masked_pos = torch.zeros(model.patch_embed.num_patches)  
    bool_masked_pos[model.patch_embed.num_patches//2:] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)
    bool_masked_pos = get_random_mask_B(model, bool_masked_pos, mask_ratio) if mask_B else bool_masked_pos # wt 
    
    return bool_masked_pos


def get_tgt_all_masked_pos(model , mask_B=False, mask_ratio=0.75):

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    else:
        model = model

    bool_masked_pos = torch.zeros(model.patch_embed.num_patches)  
    # bool_masked_pos[model.patch_embed.num_patches//2:] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)
    bool_masked_pos = get_random_mask_B(model, bool_masked_pos, mask_ratio) if mask_B else bool_masked_pos # wt 
    
    return bool_masked_pos


def reformat_output(x, tgt):
    x = x.squeeze(dim=0)
    x = torch.einsum('chw->hwc', x)

    tgt = tgt.squeeze(dim=0)
    tgt = torch.einsum('chw->hwc', tgt)

    return x.detach().numpy(), tgt.detach().numpy()



def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

