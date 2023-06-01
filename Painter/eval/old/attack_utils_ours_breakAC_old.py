from constant_utils import *

def construct_adv_A_pgd_ours_breakAC(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand'):

    model.eval()

    x = reshape(img)
    tgt = reshape(tgt)

    pos_a = torch.zeros_like(x)
    pos_a[:,:,:448,:] = 1  # A图

    pos_c = torch.zeros_like(x)
    pos_c[:,:,448:,:] = 1  # C图
 
    if rand_init: # 只保留A图的扰动
        x_adv = x.detach() + torch.from_numpy(np.random.uniform(-epsilon,
                                                                   epsilon, x.shape)).float() * pos_a
        x_adv = torch.clip(x_adv, 0.0, 1.0)
    else:
        x_adv = x.detach()

    x_adv.requires_grad_()

    bool_masked_pos = get_masked_pos(model)
    valid = torch.ones_like(tgt)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    
    bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)   # 变成True/False

    with torch.no_grad():
        ## clean c latent
        clean_c = x.detach().clone() * pos_c  # 只有C图
        clean_c_latent = model.forward_encoder(images_normalize(clean_c).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.float().to(device))
        clean_c_latent = torch.cat(clean_c_latent, dim=-1)  # [1,56,28,1024]

    for _ in range(num_steps): 
        x_adv.requires_grad_()  
        latent_adv = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.to(device))
        pred_adv = model.forward_decoder(latent_adv)
        latent_adv = torch.cat(latent_adv, dim=-1)

        loss = model.forward_loss(pred_adv, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))  ## 1.2999
        
        adv_a = x_adv.detach() * pos_a  # 只有A图
        adv_a_latent = model.forward_encoder(images_normalize(adv_a).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.float().to(device))
        adv_a_latent = torch.cat(adv_a_latent, dim=-1)  # [1,56,28,1024]

        l2_loss_adv_a_and_clean_c = L2(adv_a_latent, clean_c_latent)

        loss = loss + l2_loss_adv_a_and_clean_c
        # print(loss, l2_loss_adv_a_and_clean_c)
        loss.backward()

        # [检查] 对抗扰动的方向是tgt_adv还是x_adv
        grad_sign = x_adv.grad.detach().sign()
        perturbation = step_size * grad_sign * pos_a
        x_adv = x_adv.detach() + perturbation
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)  # [检查] 对抗样本的范围是tgt还是x
        x_adv = torch.clip(x_adv, 0.0, 1.0) 

    return reformat_output(x_adv, tgt)


def construct_adv_A_pgd_ours_breakAC_using_AA(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand'):

    ## 让A和A'分布变化, AC有个分布,如果A和A'分布不一样,那么A'C的分布也会变化

    model.eval()

    x = reshape(img)
    tgt = reshape(tgt)

    pos_a = torch.zeros_like(x)
    pos_a[:,:,:448,:] = 1  # A图

    pos_c = torch.zeros_like(x)
    pos_c[:,:,448:,:] = 1  # C图
 
    if rand_init: # 只保留A图的扰动
        x_adv = x.detach() + torch.from_numpy(np.random.uniform(-epsilon,
                                                                   epsilon, x.shape)).float() * pos_a
        x_adv = torch.clip(x_adv, 0.0, 1.0)
    else:
        x_adv = x.detach()

    x_adv.requires_grad_()

    bool_masked_pos = get_masked_pos(model)
    valid = torch.ones_like(tgt)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    
    bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)   # 变成True/False

    with torch.no_grad():
        ## clean c latent
        clean = x.detach().clone()
        clean_latent = model.forward_encoder(images_normalize(clean).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.float().to(device))
        clean_latent = torch.cat(clean_latent, dim=-1)  # [1,56,28,1024]

    for _ in range(num_steps): 
        x_adv.requires_grad_()  
        latent_adv = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.to(device))
        pred_adv = model.forward_decoder(latent_adv)
        latent_adv = torch.cat(latent_adv, dim=-1)

        loss = model.forward_loss(pred_adv, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))  ## 1.2999
        
        # adv_a = x_adv.detach() * pos_a  # 只有A图
        # adv_a_latent = model.forward_encoder(images_normalize(adv_a).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.float().to(device))
        # adv_a_latent = torch.cat(adv_a_latent, dim=-1)  # [1,56,28,1024]

        l2_loss_adv_and_clean = L2(latent_adv, clean_latent)
        # kl_loss_adv_and_clean = criterion(F.log_softmax(latent_adv, dim=-1), F.softmax(clean_latent, dim=-1))

        # print(loss, l2_loss_adv_and_clean, kl_loss_adv_and_clean)
        loss = loss + l2_loss_adv_and_clean
        loss.backward()

        # [检查] 对抗扰动的方向是tgt_adv还是x_adv
        grad_sign = x_adv.grad.detach().sign()
        perturbation = step_size * grad_sign * pos_a
        x_adv = x_adv.detach() + perturbation
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)  # [检查] 对抗样本的范围是tgt还是x
        x_adv = torch.clip(x_adv, 0.0, 1.0) 

    return reformat_output(x_adv, tgt)


def construct_adv_B_pgd_ours_breakAC(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand'):

    model.eval()

    x = reshape(img)
    tgt = reshape(tgt)

    pos_b = torch.zeros_like(tgt)
    pos_b[:,:,:448,:] = 1  # B图, 只保留前半部分 

    if rand_init:  # [检查] 是选取x还是tgt作为初始值, A和C为x, B为tgt
        tgt_adv = tgt.detach() + torch.from_numpy(np.random.uniform(-epsilon,
                                                                   epsilon, tgt.shape)).float() * pos_b
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

        loss = model.forward_loss(pred_adv, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))  ## 1.2999
        # print(loss)
        loss.backward()

        # [检查] 对抗扰动的方向是tgt_adv还是x_adv
        grad_sign = tgt_adv.grad.detach().sign()
        perturbation = step_size * grad_sign * pos_b
        tgt_adv = tgt_adv.detach() + perturbation
        tgt_adv = torch.min(torch.max(tgt_adv, tgt - epsilon), tgt + epsilon)  # [检查] 对抗样本的范围是tgt还是x
        tgt_adv = torch.clip(tgt_adv, 0.0, 1.0)   # L2 bound是指所有像素加起来不能超过budget, 而Linf bound是指每个像素的变化不能超过budget

    return reformat_output(x, tgt_adv)





def construct_adv_AB_pgd_ours_iterat(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand'):

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
        x_adv.requires_grad_()   # [检查] 进入模型的对抗样本是tgt_adv还是x_adv
        tgt_adv.requires_grad_()

        ## 根据干净的B生成A'
        latent_adv = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.to(device))
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

        
        ## 根据A'生成B'

        grad_sign_tgt = tgt_adv.grad.detach().sign()
        perturbation_tgt = step_size * grad_sign_tgt * pos_b
        tgt_adv = tgt_adv.detach() + perturbation_tgt
        tgt_adv = torch.min(torch.max(tgt_adv, tgt - epsilon), tgt + epsilon)  # [检查] 对抗样本的范围是tgt还是x
        tgt_adv = torch.clip(tgt_adv, 0.0, 1.0)   # L2 bound是指所有像素加起来不能超过budget, 而Linf bound是指每个像素的变化不能超过budget

    return reformat_output(x_adv, tgt_adv)





def construct_adv_C_pgd_our_1(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand'):

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

    with torch.no_grad():
        clean_a = x.detach().clone()
        clean_latent = model.forward_encoder(images_normalize(clean_a).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.float().to(device))
        clean_latent = torch.cat(clean_latent, dim=-1)  # [1,56,28,1024]

    for i in range(num_steps): 
        x_adv.requires_grad_()
        latent_adv = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.to(device))
        pred_adv = model.forward_decoder(latent_adv)
        latent_adv = torch.cat(latent_adv, dim=-1)
        # prob_dist2 = F.softmax(latent_adv, dim=-1)

        

        l2_loss = L2(clean_latent, latent_adv)

        # with torch.enable_grad():
        loss = model.forward_loss(pred_adv, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))  ## 1.2999
        print(loss, l2_loss)
        loss = loss + l2_loss

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


def construct_adv_C_pgd_our(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand'):

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

    with torch.no_grad():
        clean_a = x.detach().clone()
        clean_latent = model.forward_encoder(images_normalize(clean_a).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.float().to(device))
        clean_latent = torch.cat(clean_latent, dim=-1)  # [1,56,28,1024]

    for i in range(num_steps): 
        x_adv.requires_grad_()
        latent_adv = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.to(device))
        pred_adv = model.forward_decoder(latent_adv)
        latent_adv = torch.cat(latent_adv, dim=-1)
        # prob_dist2 = F.softmax(latent_adv, dim=-1)

        adv_c = x_adv.detach() * mask
        adv_latent = model.forward_encoder(images_normalize(adv_c).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.float().to(device))
        adv_latent = torch.cat(adv_latent, dim=-1)  # [1,56,28,1024]

        l2_loss = L2(clean_latent, adv_latent)

        # with torch.enable_grad():
        loss = model.forward_loss(pred_adv, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))  ## 1.2999
        # print(loss, l2_loss)
        loss = loss + l2_loss

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





def construct_adv_AB_pgd_our(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand'):

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

    with torch.no_grad():
        clean_a = x.detach().clone() * mask_x
        clean_b = tgt.detach().clone() * mask_tgt
        clean_latent = model.forward_encoder(images_normalize(clean_a).float().to(device), images_normalize(clean_b).float().to(device), bool_masked_pos.float().to(device))
        clean_latent = torch.cat(clean_latent, dim=-1)

    for i in range(num_steps): 
        x_adv.requires_grad_()   # [检查] 进入模型的对抗样本是tgt_adv还是x_adv
        tgt_adv.requires_grad_()
        latent_adv = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt_adv).float().to(device), bool_masked_pos.to(device))
        pred_adv = model.forward_decoder(latent_adv)
        latent_adv = torch.cat(latent_adv, dim=-1)
        # prob_dist2 = F.softmax(latent_adv, dim=-1)

        adv_a = x_adv.detach() * mask_x
        latent_adv_ab = model.forward_encoder(images_normalize(adv_a).float().to(device), images_normalize(tgt_adv).float().to(device), bool_masked_pos.to(device))
        latent_adv_ab = torch.cat(latent_adv_ab, dim=-1)

        l2_loss = L2(clean_latent, latent_adv_ab)

        # with torch.enable_grad():
        loss = model.forward_loss(pred_adv, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))  ## 1.2999
        # print(loss, l2_loss)
        loss = loss + l2_loss
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




def construct_adv_AC_pgd_our(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand'):   ## 42M

    model.eval()

    x = reshape(img)
    tgt = reshape(tgt)

    pos_a = torch.zeros_like(x)   # 保留pos_A的位置
    pos_a[:,:,:448,:] = 1   
    
    pos_c = torch.zeros_like(x)   # 保留pos_C的位置
    pos_c[:,:,448:,:] = 1
 
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

        adv_a = x_adv.detach() * pos_a
        adv_a_latent = model.forward_encoder(images_normalize(adv_a).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.float().to(device))
        adv_a_latent = torch.cat(adv_a_latent, dim=-1)  # [1,56,28,1024]

        adv_c = x_adv.detach() * pos_c
        adv_c_latent = model.forward_encoder(images_normalize(adv_c).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.float().to(device))
        adv_c_latent = torch.cat(adv_c_latent, dim=-1)  # [1,56,28,1024]

        l2_loss = L2(adv_a_latent, adv_c_latent)

        # with torch.enable_grad():
        loss = model.forward_loss(pred_adv, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))  ## 1.2999
        # print(loss, l2_loss)

        loss = loss + l2_loss
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


def construct_adv_B_pgd_ours_breakAB_with_BB(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand'):

    model.eval()

    x = reshape(img)
    tgt = reshape(tgt)

    pos_b = torch.zeros_like(tgt)
    pos_b[:,:,:448,:] = 1  # B图, 只保留前半部分 

    if rand_init:  # [检查] 是选取x还是tgt作为初始值, A和C为x, B为tgt
        tgt_adv = tgt.detach() + torch.from_numpy(np.random.uniform(-epsilon,
                                                                   epsilon, tgt.shape)).float() * pos_b
        tgt_adv = torch.clip(tgt_adv, 0.0, 1.0)
    else:
        tgt_adv = tgt.detach()

    tgt_adv.requires_grad_()

    bool_masked_pos = get_masked_pos(model)
    valid = torch.ones_like(tgt)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    
    bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)   # 变成True/False

    with torch.no_grad():
        ## clean c latent
        clean_b = tgt.detach().clone()
        clean_b_latent = model.forward_encoder(images_normalize(x).float().to(device), images_normalize(clean_b).float().to(device), bool_masked_pos.float().to(device))
        clean_b_latent = torch.cat(clean_b_latent, dim=-1)  # [1,56,28,1024]


    for i in range(num_steps): 
        tgt_adv.requires_grad_()   # [检查] 进入模型的对抗样本是tgt_adv还是x_adv
        latent_adv = model.forward_encoder(images_normalize(x).float().to(device), images_normalize(tgt_adv).float().to(device), bool_masked_pos.to(device))
        pred_adv = model.forward_decoder(latent_adv)
        latent_adv = torch.cat(latent_adv, dim=-1)

        loss = model.forward_loss(pred_adv, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))  ## 1.2999
        # print(loss)

        l2_loss_adv_and_clean = L2(latent_adv, clean_b_latent)
        loss = loss + l2_loss_adv_and_clean

        loss.backward()

        # [检查] 对抗扰动的方向是tgt_adv还是x_adv
        grad_sign = tgt_adv.grad.detach().sign()
        perturbation = step_size * grad_sign * pos_b
        tgt_adv = tgt_adv.detach() + perturbation
        tgt_adv = torch.min(torch.max(tgt_adv, tgt - epsilon), tgt + epsilon)  # [检查] 对抗样本的范围是tgt还是x
        tgt_adv = torch.clip(tgt_adv, 0.0, 1.0)   # L2 bound是指所有像素加起来不能超过budget, 而Linf bound是指每个像素的变化不能超过budget

    return reformat_output(x, tgt_adv)


def get_adv_img_adv_tgt_our_breakAC(img, tgt, model_painter, device, attack_id, attack_method, epsilon, num_steps):
    if attack_id == 'attack_A':
        if attack_method == 'PGD':
            # adv_img, adv_tgt = construct_adv_A_pgd_ours_breakAC(img, tgt, model_painter, device, epsilon=epsilon/255., num_steps=num_steps, step_size=2/255, rand_init='rand')
            adv_img, adv_tgt = construct_adv_A_pgd_ours_breakAC_using_AA(img, tgt, model_painter, device, epsilon=epsilon/255., num_steps=num_steps, step_size=2/255, rand_init='rand')
    elif attack_id == 'attack_C':
        if attack_method == 'PGD':
            adv_img, adv_tgt = construct_adv_C_pgd_our(img, tgt, model_painter, device, epsilon=epsilon/255., num_steps=num_steps, step_size=2/255, rand_init='rand')
    elif attack_id == 'attack_AB':
        if attack_method == 'PGD':
            adv_img, adv_tgt = construct_adv_AB_pgd_our(img, tgt, model_painter, device, epsilon=epsilon/255., num_steps=num_steps, step_size=2/255, rand_init='rand')
    elif attack_id == 'attack_AC':
        if attack_method == 'PGD':
            adv_img, adv_tgt = construct_adv_AC_pgd_our(img, tgt, model_painter, device, epsilon=epsilon/255., num_steps=num_steps, step_size=2/255, rand_init='rand')
    elif attack_id == 'attack_B':
        if attack_method == 'PGD':
            adv_img, adv_tgt = construct_adv_B_pgd_ours_breakAB_with_BB(img, tgt, model_painter, device, epsilon=epsilon/255., num_steps=num_steps, step_size=2/255, rand_init='rand')
    return adv_img, adv_tgt