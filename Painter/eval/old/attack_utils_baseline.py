from constant_utils import *


def construct_adv_AB_pgd(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand', mask_B=False, ignore_D_loss=False, lam=1, mask_ratio=0.75):

    model.eval()

    x = reshape(img)
    tgt = reshape(tgt)

    pos_a = torch.zeros_like(x)
    pos_a[:,:,:448,:] = 1  # A图, 只保留前半部分  [检查] mask的位置是上半部分还是下半部分 上部分是[:,:,:448,:], 下部分是[:,:,448:,:]

    pos_b = torch.zeros_like(tgt)
    pos_b[:,:,:448,:] = 1   # [检查] mask的位置是上半部分还是下半部分 上部分是[:,:,:448,:], 下部分是[:,:,448:,:]
 
    if rand_init: # [检查] 是选取x还是tgt作为初始值, A和C为x, B为tgt
        x_adv = x.detach() + torch.from_numpy(np.random.uniform(-epsilon,
                                                                   epsilon, x.shape)).float() * pos_a
        x_adv = torch.clip(x_adv, 0.0, 1.0)

        tgt_adv = tgt.detach() + torch.from_numpy(np.random.uniform(-epsilon,
                                                                     epsilon, tgt.shape)).float() * pos_b
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
        model.zero_grad()
        x_adv.requires_grad_()   # [检查] 进入模型的对抗样本是tgt_adv还是x_adv
        tgt_adv.requires_grad_()
        latent_adv = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt_adv).float().to(device), bool_masked_pos.to(device))
        pred_adv = model.forward_decoder(latent_adv)
        latent_adv = torch.cat(latent_adv, dim=-1)

        loss = model.forward_loss(pred_adv, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))  ## 1.2999
        # print(loss)
        loss.backward()

        # [检查] 对抗扰动的方向是tgt_adv还是x_adv
        grad_sign_x = x_adv.grad.detach().sign()
        perturbation_x = step_size * grad_sign_x * pos_a
        x_adv = x_adv.detach() + perturbation_x
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)  # [检查] 对抗样本的范围是tgt还是x
        x_adv = torch.clip(x_adv, 0.0, 1.0)   # L2 bound是指所有像素加起来不能超过budget, 而Linf bound是指每个像素的变化不能超过budget

        grad_sign_tgt = tgt_adv.grad.detach().sign()
        perturbation_tgt = step_size * grad_sign_tgt * pos_b
        tgt_adv = tgt_adv.detach() + perturbation_tgt
        tgt_adv = torch.min(torch.max(tgt_adv, tgt - epsilon), tgt + epsilon)  # [检查] 对抗样本的范围是tgt还是x
        tgt_adv = torch.clip(tgt_adv, 0.0, 1.0)   # L2 bound是指所有像素加起来不能超过budget, 而Linf bound是指每个像素的变化不能超过budget

    return reformat_output(x_adv, tgt_adv)


def construct_adv_C_pgd(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand', mask_B=False, ignore_D_loss=False, lam=1, mask_ratio=0.75):

    # print(f'mask_B: {mask_B}, ignore_D_loss: {ignore_D_loss}')

    ## ignore_D_loss: 只算D的loss

    model.eval()

    x = reshape(img)
    tgt = reshape(tgt)

    pos_c = torch.zeros_like(x)
    pos_c[:,:,448:,:] = 1   # [检查] mask的位置是上半部分还是下半部分 上部分是[:,:,:448,:], 下部分是[:,:,448:,:]

    if rand_init:
        x_adv = x.detach() + torch.from_numpy(np.random.uniform(-epsilon,
                                                                   epsilon, x.shape)).float() * pos_c
        x_adv = torch.clip(x_adv, 0.0, 1.0)
    else:
        x_adv = x.detach()

    x_adv.requires_grad_()

    bool_masked_pos_basic = get_masked_pos(model, mask_B=False, mask_ratio=0.75) # wt

    valid = torch.ones_like(tgt)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    
    bool_masked_pos_basic = bool_masked_pos_basic.flatten(1).to(torch.bool)   # 变成True/False

    if mask_B:
        ratio = mask_ratio / 100
        # print(f'mask_ratio: {ratio}')
        bool_masked_pos_B = get_masked_pos(model, mask_B=mask_B, mask_ratio=ratio) # wt
        bool_masked_pos_B = bool_masked_pos_B.flatten(1).to(torch.bool)   # 变成True/False

    for i in range(num_steps): 
        model.zero_grad()
        x_adv.requires_grad_()
        latent_adv = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos_basic.to(device))
        pred_adv = model.forward_decoder(latent_adv)
        latent_adv = torch.cat(latent_adv, dim=-1)
        # 原始loss  D loss
        loss = model.forward_loss(pred_adv, images_normalize(tgt).float().to(device), bool_masked_pos_basic.to(device), valid.to(device))

        if mask_B:
            # 把B mask后得到的变量
            mask_B_latent = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos_B.to(device))
            mask_B_pred = model.forward_decoder(mask_B_latent)
            # mask loss
            loss_mask = model.forward_loss(mask_B_pred, images_normalize(tgt).float().to(device), bool_masked_pos_B.to(device), valid.to(device), ignore_D_loss=ignore_D_loss)  # wt
            
            # print(f'loss_mask: {loss_mask}, lam:{lam}')
            loss_mask = loss_mask / lam
            # print(f'loss_mask: {loss_mask}')

            # print(loss, loss_mask)
            loss = loss + loss_mask 

        loss.backward()

        grad_sign = x_adv.grad.detach().sign()
        perturbation = step_size * grad_sign * pos_c
        x_adv = x_adv.detach() + perturbation
        x_adv = torch.min(torch.max(x_adv, x-epsilon), x + epsilon)
        x_adv = torch.clip(x_adv, 0.0, 1.0)   # L2 bound是指所有像素加起来不能超过budget, 而Linf bound是指每个像素的变化不能超过budget

    return reformat_output(x_adv, tgt)



    
def get_adv_img_adv_tgt_baseline(img, tgt, model_painter, device, attack_id, attack_method, epsilon, num_steps, mask_B, ignore_D_loss, lam, mask_ratio):
    if attack_id == 'attack_C':
        if attack_method == 'PGD':
            adv_img, adv_tgt = construct_adv_C_pgd(img, tgt, model_painter, device, 
                                                   epsilon=epsilon/255., num_steps=num_steps, step_size=2/255, rand_init='rand', 
                                                   mask_B=mask_B, ignore_D_loss=ignore_D_loss, lam=lam, mask_ratio=mask_ratio)
    elif attack_id == 'none':
        adv_img, adv_tgt = img, tgt

    return adv_img, adv_tgt