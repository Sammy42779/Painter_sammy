import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


imagenet_mean_ts=torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
imagenet_std_ts=torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]

images_normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


# criterion = torch.nn.KLDivLoss(reduction='batchmean')

L2 = torch.nn.MSELoss()


mean_pool = nn.AvgPool2d(2)


def reshape(input):
    out = torch.tensor(input)
    out = out.unsqueeze(dim=0)
    out = torch.einsum('nhwc->nchw', out)

    return out



def get_masked_pos(model):

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    else:
        model = model

    bool_masked_pos = torch.zeros(model.patch_embed.num_patches)  
    bool_masked_pos[model.patch_embed.num_patches//2:] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)
    
    return bool_masked_pos



def reformat_output(x, tgt):
    x = x.squeeze(dim=0)
    x = torch.einsum('chw->hwc', x)

    tgt = tgt.squeeze(dim=0)
    tgt = torch.einsum('chw->hwc', tgt)

    return x.detach().numpy(), tgt.detach().numpy()