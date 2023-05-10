import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from constant_utils import *

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






def construct_adv_B_pgd(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand', is_ours=False):

    model.eval()

    x = reshape(img)
    tgt = reshape(tgt)

    mask = torch.zeros_like(x)
    mask[:,:,:448,:] = 1  # B图, 只保留前半部分  [检查] mask的位置是上半部分还是下半部分 上部分是[:,:,:448,:], 下部分是[:,:,448:,:]

    ## --- 如果是我们的,就需要计算中间层,用于后续对比KL loss
    if is_ours:
        mask_only_a = torch.zeros_like(x)
        mask_only_a[:,:,:448,:] = 1  ## 用于只保留A图

        mask_only_c = torch.zeros_like(x)
        mask_only_c[:,:,448:,:] = 1  

        mask_tgt = torch.zeros_like(tgt)   ## 把B图全部mask掉


    if rand_init:  # [检查] 是选取x还是tgt作为初始值, A和C为x, B为tgt
        tgt_adv = tgt.detach() + torch.from_numpy(np.random.uniform(-epsilon,
                                                                   epsilon, tgt.shape)).float() * mask
        tgt_adv = torch.clip(tgt_adv, 0.0, 1.0)
    else:
        tgt_adv = tgt.detach()

    tgt_adv.requires_grad_()

    bool_masked_pos = get_masked_pos(model)
    valid = torch.ones_like(tgt)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    
    bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)   # 变成True/False

    for i in range(num_steps): 
        tgt_adv.requires_grad_()   # [检查] 进入模型的对抗样本是tgt_adv还是x_adv
        latent_adv = model.forward_encoder(images_normalize(x).float().to(device), images_normalize(tgt_adv).float().to(device), bool_masked_pos.to(device))
        pred_adv = model.forward_decoder(latent_adv)
        latent_adv = torch.cat(latent_adv, dim=-1)
        # prob_dist2 = F.softmax(latent_adv, dim=-1)

        # with torch.enable_grad():
        loss = model.forward_loss(pred_adv, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))  ## 1.2999
        # print(loss)
        loss.backward()

        # [检查] 对抗扰动的方向是tgt_adv还是x_adv
        grad_sign = tgt_adv.grad.detach().sign()
        perturbation = step_size * grad_sign * mask
        tgt_adv = tgt_adv.detach() + perturbation
        tgt_adv = torch.min(torch.max(tgt_adv, tgt - epsilon), tgt + epsilon)  # [检查] 对抗样本的范围是tgt还是x
        tgt_adv = torch.clip(tgt_adv, 0.0, 1.0)   # L2 bound是指所有像素加起来不能超过budget, 而Linf bound是指每个像素的变化不能超过budget

    x = x.squeeze(dim=0)
    x = torch.einsum('chw->hwc', x)

    tgt_adv = tgt_adv.squeeze(dim=0)
    tgt_adv = torch.einsum('chw->hwc', tgt_adv)

    return x.detach().numpy(), tgt_adv.detach().numpy()



def construct_adv_A_pgd(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand', is_ours=False):

    model.eval()

    x = reshape(img)
    tgt = reshape(tgt)

    mask = torch.zeros_like(x)
    mask[:,:,:448,:] = 1  # A图, 只保留前半部分  [检查] mask的位置是上半部分还是下半部分 上部分是[:,:,:448,:], 下部分是[:,:,448:,:]

    ## --- 如果是我们的,就需要计算中间层,用于后续对比KL loss
    if is_ours:
        mask_only_a = torch.zeros_like(x)
        mask_only_a[:,:,:448,:] = 1  ## 用于只保留A图

        mask_only_c = torch.zeros_like(x)
        mask_only_c[:,:,448:,:] = 1  

        mask_tgt = torch.zeros_like(tgt)   ## 把B图全部mask掉
 
    if rand_init: # [检查] 是选取x还是tgt作为初始值, A和C为x, B为tgt
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


    if is_ours:
        with torch.no_grad():  ## ? 这部分的loss需要require_grad吗
            
            # masked_tgt = tgt.detach() * mask_tgt

            # only_clean_a = x.detach() * mask_only_a

            # only_clean_a_latent = model.forward_encoder(images_normalize(only_clean_a).float().to(device), images_normalize(mask_tgt).float().to(device), bool_masked_pos.to(device))
            # only_clean_a_latent = torch.cat(only_clean_a_latent, dim=-1)

            only_c = x.detach() * mask_only_c   ## 整张图只保留左下角的C

            only_c_latent = model.forward_encoder(images_normalize(only_c).float().to(device), images_normalize(mask_tgt).float().to(device), bool_masked_pos.to(device))
            # only_c_latent = torch.cat(only_c_latent, dim=-1)
            only_c_pred = model.forward_decoder(only_c_latent)

            # clean_ac_kl = criterion(only_clean_a_latent.log_softmax(dim=-1), only_c_latent.softmax(dim=-1))



    for i in range(num_steps): 
        x_adv.requires_grad_()   # [检查] 进入模型的对抗样本是tgt_adv还是x_adv
        latent_adv = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.to(device))
        pred_adv = model.forward_decoder(latent_adv)
        latent_adv = torch.cat(latent_adv, dim=-1)
        # prob_dist2 = F.softmax(latent_adv, dim=-1)

        if is_ours:
            only_a_adv = x_adv.detach() * mask_only_a   ## 整张图只保留左上角的A_adv
            assert  (x_adv.detach() * mask_only_c).all() == only_c.all()    ## 因此C没被扰动, 所以两张img*mask后的结果应该是一致的,就是C图
            ## 进模型只有A_adv图
            only_a_adv_latent = model.forward_encoder(images_normalize(only_a_adv).float().to(device), images_normalize(mask_tgt).float().to(device), bool_masked_pos.to(device))
            # only_a_adv_latent = torch.cat(only_a_adv_latent, dim=-1)


            ### KL loss between (A_adv, C)
            kl_loss_A_adv_and_C = criterion(only_a_adv_latent.log_softmax(dim=-1), only_c_latent.softmax(dim=-1)) * 1

        # with torch.enable_grad():
        loss1 = model.forward_loss(pred_adv, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))  ## 1.2999
        # print(loss)
        if is_ours:
            loss = loss1 + kl_loss_A_adv_and_C
            print(loss1, kl_loss_A_adv_and_C)
        else:
            loss = loss1
        loss.backward()

        # [检查] 对抗扰动的方向是tgt_adv还是x_adv
        grad_sign = x_adv.grad.detach().sign()
        perturbation = step_size * grad_sign * mask
        x_adv = x_adv.detach() + perturbation
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)  # [检查] 对抗样本的范围是tgt还是x
        x_adv = torch.clip(x_adv, 0.0, 1.0)   # L2 bound是指所有像素加起来不能超过budget, 而Linf bound是指每个像素的变化不能超过budget

    x_adv = x_adv.squeeze(dim=0)
    x_adv = torch.einsum('chw->hwc', x_adv)

    tgt = tgt.squeeze(dim=0)
    tgt = torch.einsum('chw->hwc', tgt)

    return x_adv.detach().numpy(), tgt.detach().numpy()



def construct_adv_C_pgd(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand'):

    model.eval()

    x = reshape(img)
    tgt = reshape(tgt)

    mask = torch.zeros_like(x)
    mask[:,:,448:,:] = 1   # [检查] mask的位置是上半部分还是下半部分 上部分是[:,:,:448,:], 下部分是[:,:,448:,:]

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

    for i in range(num_steps): 
        x_adv.requires_grad_()
        latent_adv = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.to(device))
        pred_adv = model.forward_decoder(latent_adv)
        latent_adv = torch.cat(latent_adv, dim=-1)
        # prob_dist2 = F.softmax(latent_adv, dim=-1)

        # with torch.enable_grad():
        loss = model.forward_loss(pred_adv, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))  ## 1.2999
        # print(loss)
        loss.backward()

        grad_sign = x_adv.grad.detach().sign()
        perturbation = step_size * grad_sign * mask
        x_adv = x_adv.detach() + perturbation
        x_adv = torch.min(torch.max(x_adv, x-epsilon), x + epsilon)
        x_adv = torch.clip(x_adv, 0.0, 1.0)   # L2 bound是指所有像素加起来不能超过budget, 而Linf bound是指每个像素的变化不能超过budget

    x_adv = x_adv.squeeze(dim=0)
    x_adv = torch.einsum('chw->hwc', x_adv)

    tgt = tgt.squeeze(dim=0)
    tgt = torch.einsum('chw->hwc', tgt)

    return x_adv.detach().numpy(), tgt.detach().numpy()


def construct_adv_AB_pgd(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand', is_ours=False):

    # print('construct_adv_AB_pgd')

    model.eval()

    x = reshape(img)
    tgt = reshape(tgt)

    mask_x = torch.zeros_like(x)
    mask_x[:,:,:448,:] = 1  # A图, 只保留前半部分  [检查] mask的位置是上半部分还是下半部分 上部分是[:,:,:448,:], 下部分是[:,:,448:,:]

    mask_tgt = torch.zeros_like(tgt)
    mask_tgt[:,:,:448,:] = 1   # [检查] mask的位置是上半部分还是下半部分 上部分是[:,:,:448,:], 下部分是[:,:,448:,:]
 
    if rand_init: # [检查] 是选取x还是tgt作为初始值, A和C为x, B为tgt
        x_adv = x.detach() + torch.from_numpy(np.random.uniform(-epsilon,
                                                                   epsilon, x.shape)).float() * mask_x
        x_adv = torch.clip(x_adv, 0.0, 1.0)

        tgt_adv = tgt.detach() + torch.from_numpy(np.random.uniform(-epsilon,
                                                                     epsilon, tgt.shape)).float() * mask_tgt
        tgt_adv = torch.clip(tgt_adv, 0.0, 1.0)
    else:
        x_adv = x.detach()
        tgt_adv = tgt.detach()

    x_adv.requires_grad_()
    tgt_adv.requires_grad_()

    bool_masked_pos = get_masked_pos(model)
    valid = torch.ones_like(tgt)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    
    bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)   # 变成True/False

    for i in range(num_steps): 
        x_adv.requires_grad_()   # [检查] 进入模型的对抗样本是tgt_adv还是x_adv
        tgt_adv.requires_grad_()
        latent_adv = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt_adv).float().to(device), bool_masked_pos.to(device))
        pred_adv = model.forward_decoder(latent_adv)
        latent_adv = torch.cat(latent_adv, dim=-1)
        # prob_dist2 = F.softmax(latent_adv, dim=-1)

        # with torch.enable_grad():
        loss = model.forward_loss(pred_adv, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))  ## 1.2999
        # print(loss)
        loss.backward()

        # [检查] 对抗扰动的方向是tgt_adv还是x_adv
        grad_sign_x = x_adv.grad.detach().sign()
        perturbation_x = step_size * grad_sign_x * mask_x
        x_adv = x_adv.detach() + perturbation_x
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)  # [检查] 对抗样本的范围是tgt还是x
        x_adv = torch.clip(x_adv, 0.0, 1.0)   # L2 bound是指所有像素加起来不能超过budget, 而Linf bound是指每个像素的变化不能超过budget

        grad_sign_tgt = tgt_adv.grad.detach().sign()
        perturbation_tgt = step_size * grad_sign_tgt * mask_tgt
        tgt_adv = tgt_adv.detach() + perturbation_tgt
        tgt_adv = torch.min(torch.max(tgt_adv, tgt - epsilon), tgt + epsilon)  # [检查] 对抗样本的范围是tgt还是x
        tgt_adv = torch.clip(tgt_adv, 0.0, 1.0)   # L2 bound是指所有像素加起来不能超过budget, 而Linf bound是指每个像素的变化不能超过budget

    x_adv = x_adv.squeeze(dim=0)
    x_adv = torch.einsum('chw->hwc', x_adv)

    tgt_adv = tgt_adv.squeeze(dim=0)
    tgt_adv = torch.einsum('chw->hwc', tgt_adv)

    return x_adv.detach().numpy(), tgt_adv.detach().numpy()



def construct_adv_AC_pgd(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand'):

    model.eval()

    x = reshape(img)
    tgt = reshape(tgt)
 
    if rand_init: # [检查] 是选取x还是tgt作为初始值, A和C为x, B为tgt
        x_adv = x.detach() + torch.from_numpy(np.random.uniform(-epsilon,
                                                                   epsilon, x.shape)).float()
        x_adv = torch.clip(x_adv, 0.0, 1.0)
    else:
        x_adv = x.detach()

    x_adv.requires_grad_()

    bool_masked_pos = get_masked_pos(model)
    valid = torch.ones_like(tgt)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    
    bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)   # 变成True/False

    for i in range(num_steps): 
        x_adv.requires_grad_()   # [检查] 进入模型的对抗样本是tgt_adv还是x_adv
        latent_adv = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.to(device))
        pred_adv = model.forward_decoder(latent_adv)
        latent_adv = torch.cat(latent_adv, dim=-1)
        # prob_dist2 = F.softmax(latent_adv, dim=-1)

        # with torch.enable_grad():
        loss = model.forward_loss(pred_adv, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))  ## 1.2999
        # print(loss)
        loss.backward()

        # [检查] 对抗扰动的方向是tgt_adv还是x_adv
        grad_sign = x_adv.grad.detach().sign()
        perturbation = step_size * grad_sign 
        x_adv = x_adv.detach() + perturbation
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)  # [检查] 对抗样本的范围是tgt还是x
        x_adv = torch.clip(x_adv, 0.0, 1.0)   # L2 bound是指所有像素加起来不能超过budget, 而Linf bound是指每个像素的变化不能超过budget

    x_adv = x_adv.squeeze(dim=0)
    x_adv = torch.einsum('chw->hwc', x_adv)

    tgt = tgt.squeeze(dim=0)
    tgt = torch.einsum('chw->hwc', tgt)

    return x_adv.detach().numpy(), tgt.detach().numpy()



def construct_adv_BC_pgd(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand'):

    print('construct_adv_BC_pgd')

    model.eval()

    x = reshape(img)
    tgt = reshape(tgt)

    mask_x = torch.zeros_like(x)
    mask_x[:,:,448:,:] = 1  # A图, 只保留前半部分  [检查] mask的位置是上半部分还是下半部分 上部分是[:,:,:448,:], 下部分是[:,:,448:,:]

    mask_tgt = torch.zeros_like(tgt)
    mask_tgt[:,:,:448,:] = 1   # [检查] mask的位置是上半部分还是下半部分 上部分是[:,:,:448,:], 下部分是[:,:,448:,:]
 
    if rand_init: # [检查] 是选取x还是tgt作为初始值, A和C为x, B为tgt
        x_adv = x.detach() + torch.from_numpy(np.random.uniform(-epsilon,
                                                                   epsilon, x.shape)).float() * mask_x
        x_adv = torch.clip(x_adv, 0.0, 1.0)

        tgt_adv = tgt.detach() + torch.from_numpy(np.random.uniform(-epsilon,
                                                                     epsilon, tgt.shape)).float() * mask_tgt
        tgt_adv = torch.clip(tgt_adv, 0.0, 1.0)
    else:
        x_adv = x.detach()
        tgt_adv = tgt.detach()

    x_adv.requires_grad_()
    tgt_adv.requires_grad_()

    bool_masked_pos = get_masked_pos(model)
    valid = torch.ones_like(tgt)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    
    bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)   # 变成True/False

    for i in range(num_steps): 
        x_adv.requires_grad_()   # [检查] 进入模型的对抗样本是tgt_adv还是x_adv
        tgt_adv.requires_grad_()
        latent_adv = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt_adv).float().to(device), bool_masked_pos.to(device))
        pred_adv = model.forward_decoder(latent_adv)
        latent_adv = torch.cat(latent_adv, dim=-1)
        # prob_dist2 = F.softmax(latent_adv, dim=-1)

        # with torch.enable_grad():
        loss = model.forward_loss(pred_adv, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))  ## 1.2999
        # print(loss)
        loss.backward()

        # [检查] 对抗扰动的方向是tgt_adv还是x_adv
        grad_sign_x = x_adv.grad.detach().sign()
        perturbation_x = step_size * grad_sign_x * mask_x
        x_adv = x_adv.detach() + perturbation_x
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)  # [检查] 对抗样本的范围是tgt还是x
        x_adv = torch.clip(x_adv, 0.0, 1.0)   # L2 bound是指所有像素加起来不能超过budget, 而Linf bound是指每个像素的变化不能超过budget

        grad_sign_tgt = tgt_adv.grad.detach().sign()
        perturbation_tgt = step_size * grad_sign_tgt * mask_tgt
        tgt_adv = tgt_adv.detach() + perturbation_tgt
        tgt_adv = torch.min(torch.max(tgt_adv, tgt - epsilon), tgt + epsilon)  # [检查] 对抗样本的范围是tgt还是x
        tgt_adv = torch.clip(tgt_adv, 0.0, 1.0)   # L2 bound是指所有像素加起来不能超过budget, 而Linf bound是指每个像素的变化不能超过budget

    x_adv = x_adv.squeeze(dim=0)
    x_adv = torch.einsum('chw->hwc', x_adv)

    tgt_adv = tgt_adv.squeeze(dim=0)
    tgt_adv = torch.einsum('chw->hwc', tgt_adv)

    return x_adv.detach().numpy(), tgt_adv.detach().numpy()






# def construct_adv_A_pgd(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand'):

#     loss_list = []

#     model.eval()

#     x = reshape(img)
#     tgt = reshape(tgt)

#     if rand_init:
#         delta_a = torch.from_numpy(np.random.uniform(-epsilon, epsilon, x[:,:,:448,:].shape)).float()
#         delta_a = torch.clip(x[:,:,:448,:] + delta_a, 0.0, 1.0) - x[:,:,:448,:]
#     else:
#         delta_a = torch.zeros_like(x[:,:,:448,:])

#     x_adv = x.clone().detach()

#     bool_masked_pos = get_masked_pos(model)
#     valid = torch.ones_like(tgt)
#     if isinstance(model, torch.nn.parallel.DistributedDataParallel):
#         model = model.module

#     for i in range(num_steps):
#         delta_a.requires_grad_()   # [1,3,448,448]
        
#         x_adv[:,:,:448,:] += delta_a  # A图

#         bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)   # 变成True/False
#         latent = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.to(device))
#         pred = model.forward_decoder(latent)
#         model.zero_grad()
#         with torch.enable_grad():
#             loss = model.forward_loss(pred, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))
        
#         # loss_list.append(loss.detach().item())    
#         if i == 0:
#             loss.backward(retain_graph=True)
#         else:
#             loss.backward()

#         grad_sign_a = delta_a.grad.detach().sign()  # [1,3,448,448] 代表方向
#         perturbation_a = step_size * grad_sign_a
#         delta_a = delta_a + perturbation_a  # update delta
#         delta_a = torch.clip(delta_a, -epsilon, epsilon)
#         delta_a = torch.clip(x[:,:,:448,:] + delta_a, 0.0, 1.0) - x[:,:,:448,:]
    
    
#     x[:,:,:448,:] += delta_a  ## 最终的对抗样本是在原图上加上delta_a
#     x = x.squeeze(dim=0)
#     x = torch.einsum('chw->hwc', x)

#     tgt = tgt.squeeze(dim=0)
#     tgt = torch.einsum('chw->hwc', tgt)

#     return x.detach().numpy(), tgt.detach().numpy(), loss_list









def get_adv_img_adv_tgt(img, tgt, model_painter, device, attack_id, attack_method, epsilon, num_steps, is_ours=False):
    if attack_id == 'attack_A':
        if attack_method == 'PGD':
            adv_img, adv_tgt = construct_adv_A_pgd(img, tgt, model_painter, device, epsilon=epsilon/255., num_steps=num_steps, step_size=2/255, rand_init='rand', is_ours=is_ours)
    elif attack_id == 'attack_B':
        if attack_method == 'PGD':
            adv_img, adv_tgt = construct_adv_B_pgd(img, tgt, model_painter, device, epsilon=epsilon/255., num_steps=num_steps, step_size=2/255, rand_init='rand', is_ours=is_ours)
    elif attack_id == 'attack_C':
        if attack_method == 'PGD':
            adv_img, adv_tgt = construct_adv_C_pgd(img, tgt, model_painter, device, epsilon=epsilon/255., num_steps=num_steps, step_size=2/255, rand_init='rand', is_ours=is_ours)
    elif attack_id == 'attack_AB':
        if attack_method == 'PGD':
            adv_img, adv_tgt = construct_adv_AB_pgd(img, tgt, model_painter, device, epsilon=epsilon/255., num_steps=num_steps, step_size=2/255, rand_init='rand', is_ours=is_ours)
    elif attack_id == 'attack_AC':
        if attack_method == 'PGD':
            adv_img, adv_tgt = construct_adv_AC_pgd(img, tgt, model_painter, device, epsilon=epsilon/255., num_steps=num_steps, step_size=2/255, rand_init='rand', is_ours=is_ours)
    # elif attack_id == 'attack_BC':
    #     if attack_method == 'FGSM':
    #         adv_img, adv_tgt = construct_adv_BC_fgsm(img, tgt, model_painter, device, alpha=epsilon/255., rand_init=True)
    # elif attack_id == 'attack_ABC':
    #     if attack_method == 'FGSM':
    #         adv_img, adv_tgt = construct_adv_ABC_fgsm(img, tgt, model_painter, device, alpha=epsilon/255., rand_init=True)
    elif attack_id == 'none':
        adv_img, adv_tgt = img, tgt

    return adv_img, adv_tgt
    

