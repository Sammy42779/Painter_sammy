import torch
import numpy as np

from constant_utils import imagenet_mean, imagenet_std

def reshape(input):
    out = torch.tensor(input)
    out = out.unsqueeze(dim=0)
    out = torch.einsum('nhwc->nchw', out)

    return out

def get_masked_pos(model):
    bool_masked_pos = torch.zeros(model.patch_embed.num_patches)
    bool_masked_pos[model.patch_embed.num_patches//2:] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)
    
    return bool_masked_pos


def construct_adv_AC_fgsm(img, tgt, model, device, alpha=0.031, rand_init=False):
    model.eval()

    imagenet_mean_ts=torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    imagenet_std_ts=torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]

    x = reshape(img)
    tgt = reshape(tgt)

    if rand_init:
        img_adv = x.detach() + torch.from_numpy(np.random.uniform(-alpha,
                                                                   alpha, x.shape)).float().to(device)
        img_adv = torch.clip((img_adv * imagenet_std + imagenet_mean), 0.0, 1.0)
    else:
        img_adv = x.detach()

    img_adv.requires_grad_()

    bool_masked_pos = get_masked_pos(model)
    valid = torch.ones_like(tgt)

    model.zero_grad()
    with torch.enable_grad():
        loss, y, mask = model(img_adv.float().to(device), tgt.float().to(device), bool_masked_pos.to(device), valid.float().to(device))

    loss.backward()

    grad_sign = img_adv.grad.detach().sign()
    perturbation = alpha * grad_sign
    img_adv = img_adv + perturbation
    img_adv = torch.min(torch.max(img_adv, x-alpha), x + alpha)
    # img_adv = torch.clip((img_adv.detach() * imagenet_std_ts + imagenet_mean_ts), 0.0, 1.0)

    img_adv = img_adv.squeeze(dim=0)
    img_adv = torch.einsum('chw->hwc', img_adv)

    return img_adv.detach().numpy()


def construct_adv_B_fgsm(img, tgt, model, device, alpha=0.031, rand_init=False):
    model.eval()

    imagenet_mean_ts=torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    imagenet_std_ts=torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]

    x = reshape(img)
    tgt = reshape(tgt)

    if rand_init:
        tgt_adv = tgt.detach() + torch.from_numpy(np.random.uniform(-alpha,
                                                                   alpha, tgt.shape)).float().to(device)
        tgt_adv = torch.clip((tgt_adv * imagenet_std + imagenet_mean), 0.0, 1.0)
    else:
        tgt_adv = tgt.detach()

    tgt_adv.requires_grad_()

    bool_masked_pos = get_masked_pos(model)
    valid = torch.ones_like(tgt)

    model.zero_grad()

    # perturbation_mask = torch.zeros(x.shape)
    # perturbation_mask[]
    
    with torch.enable_grad():
        loss, y, mask = model(x.float().to(device), tgt_adv.float().to(device), bool_masked_pos.to(device), valid.float().to(device))

    loss.backward()

    grad_sign = tgt_adv.grad.detach().sign()
    perturbation = alpha * grad_sign
    tgt_adv = tgt_adv + perturbation
    tgt_adv = torch.min(torch.max(tgt_adv, tgt-alpha), tgt + alpha)
    # tgt_adv = torch.clip((tgt_adv.detach() * imagenet_std_ts + imagenet_mean_ts), 0.0, 1.0)

    tgt_adv = tgt_adv.squeeze(dim=0)
    tgt_adv = torch.einsum('chw->hwc', tgt_adv)

    return tgt_adv.detach().numpy()


def construct_adv_AC_fgsm(img, tgt, model, device, alpha=0.031, rand_init=False):
    model.eval()

    imagenet_mean_ts=torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    imagenet_std_ts=torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]

    x = reshape(img)
    tgt = reshape(tgt)

    if rand_init:
        img_adv = x.detach() + torch.from_numpy(np.random.uniform(-alpha,
                                                                   alpha, x.shape)).float().to(device)
        img_adv = torch.clip((img_adv * imagenet_std + imagenet_mean), 0.0, 1.0)
    else:
        img_adv = x.detach()

    img_adv.requires_grad_()

    bool_masked_pos = get_masked_pos(model)
    valid = torch.ones_like(tgt)

    model.zero_grad()
    with torch.enable_grad():
        loss, y, mask = model(img_adv.float().to(device), tgt.float().to(device), bool_masked_pos.to(device), valid.float().to(device))

    loss.backward()

    grad_sign = img_adv.grad.detach().sign()
    perturbation = alpha * grad_sign
    img_adv = img_adv + perturbation
    img_adv = torch.min(torch.max(img_adv, x-alpha), x + alpha)
    # img_adv = torch.clip((img_adv.detach() * imagenet_std_ts + imagenet_mean_ts), 0.0, 1.0)

    img_adv = img_adv.squeeze(dim=0)
    img_adv = torch.einsum('chw->hwc', img_adv)

    return img_adv.detach().numpy()



def construct_adv_AC_fgsm_our(img, tgt, model, device, alpha=0.031, rand_init=False):
    model.eval()

    imagenet_mean_ts=torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    imagenet_std_ts=torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]

    x = reshape(img)
    tgt = reshape(tgt)

    if rand_init:
        img_adv = x.detach() + torch.from_numpy(np.random.uniform(-alpha,
                                                                   alpha, x.shape)).float().to(device)
        img_adv = torch.clip((img_adv * imagenet_std + imagenet_mean), 0.0, 1.0)
    else:
        img_adv = x.detach()

    img_adv.requires_grad_()

    bool_masked_pos = get_masked_pos(model)
    valid = torch.ones_like(tgt)

    model.zero_grad()
    with torch.enable_grad():
        loss, y, mask = model(img_adv.float().to(device), tgt.float().to(device), bool_masked_pos.to(device), valid.float().to(device))

        # 加上B的分布要和原始的分布差别很大
        B_KL

    loss.backward()

    grad_sign = img_adv.grad.detach().sign()
    perturbation = alpha * grad_sign
    img_adv = img_adv + perturbation
    img_adv = torch.min(torch.max(img_adv, x-alpha), x + alpha)
    # img_adv = torch.clip((img_adv.detach() * imagenet_std_ts + imagenet_mean_ts), 0.0, 1.0)

    img_adv = img_adv.squeeze(dim=0)
    img_adv = torch.einsum('chw->hwc', img_adv)

    return img_adv.detach().numpy()