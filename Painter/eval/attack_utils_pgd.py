import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from constant_utils import imagenet_mean, imagenet_std, images_normalize
from attack_utils_with_clip import get_masked_pos, reshape




def construct_adv_C_pgd_our(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand'):
    model.eval()

    x = reshape(img)
    tgt = reshape(tgt)

    if rand_init:
        delta_c = torch.from_numpy(np.random.uniform(-epsilon, epsilon, x[:,:,448:,:].shape)).float()
        delta_c = torch.clip(x[:,:,448:,:] + delta_c, 0.0, 1.0) - x[:,:,448:,:]
    else:
        delta_c = torch.zeros_like(x[:,:,448:,:])

    x_adv = x.clone().detach()

    delta_c.requires_grad_()   # [1,3,448,448]

    bool_masked_pos = get_masked_pos(model)
    valid = torch.ones_like(tgt)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    x_adv[:,:,448:,:] += delta_c  # C图
    
    bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)   # 变成True/False

    with torch.no_grad():
        latent = model.forward_encoder(images_normalize(x).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.to(device))
        pred = model.forward_decoder(latent)
        latent1 = torch.cat(latent, dim=-1)

        prob_dist1 = F.softmax(latent1, dim=3)

    for i in range(num_steps): 
        latent_adv = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.to(device))
        pred_adv = model.forward_decoder(latent_adv)
        latent_adv = torch.cat(latent_adv, dim=-1)
        prob_dist2 = F.softmax(latent_adv, dim=3)

        loss1 = model.forward_loss(pred_adv, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))  ## 1.2999
        # Compute the KL divergence loss
        kl_loss = F.kl_div(prob_dist1.log(), prob_dist2, reduction='batchmean')
        loss = loss1 + kl_loss   # encourage dissimilarity between the two latent
        print(loss1, kl_loss)
        loss.backward()

        grad_sign_c = delta_c.grad.detach().sign()  # [1,3,448,448] 代表方向
        perturbation_c = step_size * grad_sign_c
        delta_c = delta_c + perturbation_c  # update delta
        delta_c = torch.clip(delta_c, -epsilon, epsilon)
        delta_c = torch.clip(x[:,:,448:,:] + delta_c, 0.0, 1.0) - x[:,:,448:,:]

    x[:,:,448:,:] += delta_c

    x = x.squeeze(dim=0)
    x = torch.einsum('chw->hwc', x)

    tgt = tgt.squeeze(dim=0)
    tgt = torch.einsum('chw->hwc', tgt)

    return x.detach().numpy(), tgt.detach().numpy()



def construct_adv_C_pgd(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand'):

    model.eval()

    x = reshape(img)
    tgt = reshape(tgt)

    mask = torch.zeros_like(x)
    mask[:,:,448:,:] = 1

    if rand_init:
        x_adv = x.detach() + torch.from_numpy(np.random.uniform(-epsilon,
                                                                   epsilon, x.shape)).float() * mask
        x_adv = torch.clip(x_adv, 0.0, 1.0)
    else:
        x_adv = x.detach()

    x_adv.requires_grad_()

    bool_masked_pos = get_masked_pos(model)
    valid = torch.ones_like(tgt)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    
    bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)   # 变成True/False

    # 干净数据的latent
    with torch.no_grad():
        latent = model.forward_encoder(images_normalize(x).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.to(device))
        pred = model.forward_decoder(latent)
        latent1 = torch.cat(latent, dim=-1)
        prob_dist1 = F.softmax(latent1, dim=-1)

    for i in range(num_steps): 
        latent_adv = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.to(device))
        pred_adv = model.forward_decoder(latent_adv)
        latent_adv = torch.cat(latent_adv, dim=-1)
        prob_dist2 = F.softmax(latent_adv, dim=-1)

        with torch.enable_grad():
            loss1 = model.forward_loss(pred_adv, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))  ## 1.2999
            kl_loss = F.kl_div(prob_dist1.log(), prob_dist2, reduction='batchmean')  # kl(prob_dist1.log_softmax(dim=-1), prob_dist2.softmax(dim=-1))
            loss = loss1 + kl_loss   # encourage dissimilarity between the two latents
            print(loss1, kl_loss)
            loss.backward()

        grad_sign = x_adv.grad.detach().sign()
        perturbation = step_size * grad_sign * mask
        x_adv = x_adv + perturbation
        x_adv = torch.min(torch.max(x_adv, x-epsilon), x + epsilon)
        x_adv = torch.clip(x_adv, 0.0, 1.0)   # L2 bound是指所有像素加起来不能超过budget, 而Linf bound是指每个像素的变化不能超过budget

    x_adv = x_adv.squeeze(dim=0)
    x_adv = torch.einsum('chw->hwc', x_adv)

    tgt = tgt.squeeze(dim=0)
    tgt = torch.einsum('chw->hwc', tgt)

    return x_adv.detach().numpy(), tgt.detach().numpy()




def pgd(model, data, label, device, epsilon, num_steps, step_size, rand_init=None):
    model.eval()
    if rand_init == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).to(device).detach()
    elif rand_init == "rand":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon,
                                                                   epsilon, data.shape)).float().to(device)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = data.detach()

    for _ in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        model.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss(reduction="mean")(output, label)
        loss.backward()

        perturbation = step_size * x_adv.grad.sign()
        x_adv = x_adv.detach() + perturbation
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv