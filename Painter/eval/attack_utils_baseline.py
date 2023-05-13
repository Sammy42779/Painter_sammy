from constant_utils import *

def construct_adv_A_pgd(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand'):

    model.eval()

    x = reshape(img)
    tgt = reshape(tgt)

    pos_a = torch.zeros_like(x)
    pos_a[:,:,:448,:] = 1  # A图
 
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

    for _ in range(num_steps): 
        x_adv.requires_grad_()  
        latent_adv = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.to(device))
        pred_adv = model.forward_decoder(latent_adv)
        latent_adv = torch.cat(latent_adv, dim=-1)

        loss = model.forward_loss(pred_adv, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))  ## 1.2999
        # print(loss)
        loss.backward()

        # [检查] 对抗扰动的方向是tgt_adv还是x_adv
        grad_sign = x_adv.grad.detach().sign()
        perturbation = step_size * grad_sign * pos_a
        x_adv = x_adv.detach() + perturbation
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)  # [检查] 对抗样本的范围是tgt还是x
        x_adv = torch.clip(x_adv, 0.0, 1.0) 

    return reformat_output(x_adv, tgt)



def construct_adv_B_pgd(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand'):

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



def construct_adv_C_pgd(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand', mask_B=False, ignore_D_loss=False):

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

    bool_masked_pos = get_masked_pos(model, mask_B=mask_B, mask_ratio=0.75) # wt
    valid = torch.ones_like(tgt)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    
    bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)   # 变成True/False

    for i in range(num_steps): 
        x_adv.requires_grad_()
        latent_adv = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.to(device))
        pred_adv = model.forward_decoder(latent_adv)
        latent_adv = torch.cat(latent_adv, dim=-1)

        loss = model.forward_loss(pred_adv, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device), ignore_D_loss=ignore_D_loss)  # wt
        # print(loss)
        loss.backward()

        grad_sign = x_adv.grad.detach().sign()
        perturbation = step_size * grad_sign * pos_c
        x_adv = x_adv.detach() + perturbation
        x_adv = torch.min(torch.max(x_adv, x-epsilon), x + epsilon)
        x_adv = torch.clip(x_adv, 0.0, 1.0)   # L2 bound是指所有像素加起来不能超过budget, 而Linf bound是指每个像素的变化不能超过budget

    return reformat_output(x_adv, tgt)



def construct_adv_AB_pgd(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand'):

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

        loss = model.forward_loss(pred_adv, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))  ## 1.2999
        # print(loss)
        loss.backward()

        # [检查] 对抗扰动的方向是tgt_adv还是x_adv
        grad_sign = x_adv.grad.detach().sign()
        perturbation = step_size * grad_sign 
        x_adv = x_adv.detach() + perturbation
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)  # [检查] 对抗样本的范围是tgt还是x
        x_adv = torch.clip(x_adv, 0.0, 1.0)   # L2 bound是指所有像素加起来不能超过budget, 而Linf bound是指每个像素的变化不能超过budget

    return reformat_output(x_adv, tgt)



def construct_adv_BC_pgd(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand'):

    model.eval()

    x = reshape(img)
    tgt = reshape(tgt)

    pos_c = torch.zeros_like(x)
    pos_c[:,:,448:,:] = 1  # C图, 只保留后半部分 

    pos_b = torch.zeros_like(tgt)
    pos_b[:,:,:448,:] = 1   # [检查] mask的位置是上半部分还是下半部分 上部分是[:,:,:448,:], 下部分是[:,:,448:,:]
 
    if rand_init: # [检查] 是选取x还是tgt作为初始值, A和C为x, B为tgt
        x_adv = x.detach() + torch.from_numpy(np.random.uniform(-epsilon,
                                                                   epsilon, x.shape)).float() * pos_c
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
        latent_adv = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt_adv).float().to(device), bool_masked_pos.to(device))
        pred_adv = model.forward_decoder(latent_adv)
        latent_adv = torch.cat(latent_adv, dim=-1)

        loss = model.forward_loss(pred_adv, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))  ## 1.2999
        # print(loss)
        loss.backward()

        # [检查] 对抗扰动的方向是tgt_adv还是x_adv
        grad_sign_x = x_adv.grad.detach().sign()
        perturbation_x = step_size * grad_sign_x * pos_c
        x_adv = x_adv.detach() + perturbation_x
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)  # [检查] 对抗样本的范围是tgt还是x
        x_adv = torch.clip(x_adv, 0.0, 1.0)   # L2 bound是指所有像素加起来不能超过budget, 而Linf bound是指每个像素的变化不能超过budget

        grad_sign_tgt = tgt_adv.grad.detach().sign()
        perturbation_tgt = step_size * grad_sign_tgt * pos_b
        tgt_adv = tgt_adv.detach() + perturbation_tgt
        tgt_adv = torch.min(torch.max(tgt_adv, tgt - epsilon), tgt + epsilon)  # [检查] 对抗样本的范围是tgt还是x
        tgt_adv = torch.clip(tgt_adv, 0.0, 1.0)   # L2 bound是指所有像素加起来不能超过budget, 而Linf bound是指每个像素的变化不能超过budget

    return reformat_output(x_adv, tgt_adv)



def construct_adv_ABC_pgd(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand'):

    model.eval()

    x = reshape(img)
    tgt = reshape(tgt)

    pos_b = torch.zeros_like(tgt)
    pos_b[:,:,:448,:] = 1   # [检查] mask的位置是上半部分还是下半部分 上部分是[:,:,:448,:], 下部分是[:,:,448:,:]
 
    if rand_init: # [检查] 是选取x还是tgt作为初始值, A和C为x, B为tgt
        x_adv = x.detach() + torch.from_numpy(np.random.uniform(-epsilon,
                                                                   epsilon, x.shape)).float() 
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
        latent_adv = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt_adv).float().to(device), bool_masked_pos.to(device))
        pred_adv = model.forward_decoder(latent_adv)
        latent_adv = torch.cat(latent_adv, dim=-1)

        loss = model.forward_loss(pred_adv, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))  ## 1.2999
        # print(loss)
        loss.backward()

        # [检查] 对抗扰动的方向是tgt_adv还是x_adv
        grad_sign_x = x_adv.grad.detach().sign()
        perturbation_x = step_size * grad_sign_x 
        x_adv = x_adv.detach() + perturbation_x
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)  # [检查] 对抗样本的范围是tgt还是x
        x_adv = torch.clip(x_adv, 0.0, 1.0)   # L2 bound是指所有像素加起来不能超过budget, 而Linf bound是指每个像素的变化不能超过budget

        grad_sign_tgt = tgt_adv.grad.detach().sign()
        perturbation_tgt = step_size * grad_sign_tgt * pos_b
        tgt_adv = tgt_adv.detach() + perturbation_tgt
        tgt_adv = torch.min(torch.max(tgt_adv, tgt - epsilon), tgt + epsilon)  # [检查] 对抗样本的范围是tgt还是x
        tgt_adv = torch.clip(tgt_adv, 0.0, 1.0)   # L2 bound是指所有像素加起来不能超过budget, 而Linf bound是指每个像素的变化不能超过budget

    return reformat_output(x_adv, tgt_adv)



def construct_adv_A_fgsm(img, tgt, model, device, alpha=0.031, rand_init=True):

    model.eval()

    x = reshape(img)
    tgt = reshape(tgt)

    if rand_init:
        delta_a = torch.from_numpy(np.random.uniform(-alpha, alpha, x[:,:,:448,:].shape)).float()
        delta_a = torch.clip(x[:,:,:448,:] + delta_a, 0.0, 1.0) - x[:,:,:448,:]
    else:
        delta_a = torch.zeros_like(x[:,:,:448,:])

    x_adv = x.clone().detach()

    delta_a.requires_grad_()   # [1,3,448,448]

    bool_masked_pos = get_masked_pos(model)
    valid = torch.ones_like(tgt)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    x_adv[:,:,:448,:] += delta_a  # A图
    
    bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)   # 变成True/False
    latent = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.to(device))
    pred = model.forward_decoder(latent)
    model.zero_grad()
    with torch.enable_grad():
        loss = model.forward_loss(pred, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))
        
    loss.backward()

    grad_sign_a = delta_a.grad.detach().sign()  # [1,3,448,448] 代表方向
    perturbation_a = alpha * grad_sign_a
    delta_a = delta_a + perturbation_a  # update delta
    delta_a = torch.clip(delta_a, -alpha, alpha)

    delta_a = torch.clip(x[:,:,:448,:] + delta_a, 0.0, 1.0) - x[:,:,:448,:]
    x[:,:,:448,:] += delta_a  ## 最终的对抗样本是在原图上加上delta_a

    x = x.squeeze(dim=0)
    x = torch.einsum('chw->hwc', x)

    tgt = tgt.squeeze(dim=0)
    tgt = torch.einsum('chw->hwc', tgt)

    return x.detach().numpy(), tgt.detach().numpy()



def construct_adv_B_fgsm(img, tgt, model, device, alpha=0.031, rand_init=True):

    # print('----------construct_adv_C_fgsm_with_clip----------')

    model.eval()

    x = reshape(img)
    tgt = reshape(tgt)

    if rand_init:
        # B图 tgt[:,:,:448,:]
        delta_b = torch.from_numpy(np.random.uniform(-alpha, alpha, tgt[:,:,:448,:].shape)).float()
        delta_b = torch.clip(tgt[:,:,:448,:] + delta_b, 0.0, 1.0) - tgt[:,:,:448,:]
    else:
        delta_b = torch.zeros_like(tgt[:,:,:448,:])

    tgt_adv = tgt.clone().detach()

    delta_b.requires_grad_()   # [1,3,448,448]

    bool_masked_pos = get_masked_pos(model)
    valid = torch.ones_like(tgt)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    tgt_adv[:,:,:448,:] += delta_b
    
    bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)   # 变成True/False
    latent = model.forward_encoder(images_normalize(x).float().to(device), images_normalize(tgt_adv).float().to(device), bool_masked_pos.to(device))
    pred = model.forward_decoder(latent)
    model.zero_grad()
    with torch.enable_grad():
        loss = model.forward_loss(pred, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))
        
    loss.backward()

    grad_sign_b = delta_b.grad.detach().sign()  # [1,3,448,448] 代表方向
    perturbation_b = alpha * grad_sign_b
    delta_b = delta_b + perturbation_b  # update delta
    delta_b = torch.clip(delta_b, -alpha, alpha)
    delta_b = torch.clip(tgt[:,:,:448,:] + delta_b, 0.0, 1.0) - tgt[:,:,:448,:]

    tgt[:,:,:448,:] += delta_b  ## 最终的对抗样本是在原图上加上delta_b

    x = x.squeeze(dim=0)
    x = torch.einsum('chw->hwc', x)

    tgt = tgt_adv.squeeze(dim=0)
    tgt = torch.einsum('chw->hwc', tgt)

    return x.detach().numpy(), tgt.detach().numpy()



def construct_adv_C_fgsm(img, tgt, model, device, alpha=0.031, rand_init=True):
    model.eval()

    x = reshape(img)
    tgt = reshape(tgt)

    if rand_init:
        delta_c = torch.from_numpy(np.random.uniform(-alpha, alpha, x[:,:,448:,:].shape)).float()
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
    latent = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.to(device))
    pred = model.forward_decoder(latent)
    model.zero_grad()
    with torch.enable_grad():
        loss = model.forward_loss(pred, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))
        
    loss.backward()

    grad_sign_c = delta_c.grad.detach().sign()  # [1,3,448,448] 代表方向
    perturbation_c = alpha * grad_sign_c
    delta_c = delta_c + perturbation_c  # update delta
    delta_c = torch.clip(delta_c, -alpha, alpha)

    delta_c = torch.clip(x[:,:,448:,:] + delta_c, 0.0, 1.0) - x[:,:,448:,:]
    x[:,:,448:,:] += delta_c

    x = x.squeeze(dim=0)
    x = torch.einsum('chw->hwc', x)

    tgt = tgt.squeeze(dim=0)
    tgt = torch.einsum('chw->hwc', tgt)

    return x.detach().numpy(), tgt.detach().numpy()



def construct_adv_AB_fgsm(img, tgt, model, device, alpha=0.031, rand_init=True):
    model.eval()

    x = reshape(img)
    tgt = reshape(tgt)

    if rand_init:
        # A图 x[:,:,:448,:]
        delta_a = torch.from_numpy(np.random.uniform(-alpha, alpha, x[:,:,:448,:].shape)).float()
        delta_a = torch.clip(x[:,:,:448,:] + delta_a, 0.0, 1.0) - x[:,:,:448,:]

        # B图 tgt[:,:,:448,:]
        delta_b = torch.from_numpy(np.random.uniform(-alpha, alpha, tgt[:,:,:448,:].shape)).float()
        delta_b = torch.clip(tgt[:,:,:448,:] + delta_b, 0.0, 1.0) - tgt[:,:,:448,:]
    else:
        delta_a = torch.zeros_like(x[:,:,:448,:])
        delta_b = torch.zeros_like(tgt[:,:,:448,:])

    x_adv = x.clone().detach()
    tgt_adv = tgt.clone().detach()

    delta_a.requires_grad_()   # [1,3,448,448]
    delta_b.requires_grad_()   # [1,3,448,448]

    bool_masked_pos = get_masked_pos(model)
    valid = torch.ones_like(tgt)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    x_adv[:,:,:448,:] += delta_a  # A和B图都加上噪声
    tgt_adv[:,:,:448,:] += delta_b
    
    model.zero_grad()
    bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)   # 变成True/False
    latent = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt_adv).float().to(device), bool_masked_pos.to(device))
    pred = model.forward_decoder(latent)
    with torch.enable_grad():
        loss = model.forward_loss(pred, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))
        
    loss.backward()

    grad_sign_a = delta_a.grad.detach().sign()  # [1,3,448,448] 代表方向
    perturbation_a = alpha * grad_sign_a
    delta_a = delta_a + perturbation_a  # update delta
    delta_a = torch.clip(delta_a, -alpha, alpha)
    delta_a = torch.clip(x[:,:,:448,:] + delta_a, 0.0, 1.0) - x[:,:,:448,:]

    grad_sign_b = delta_b.grad.detach().sign()  # [1,3,448,448] 代表方向
    perturbation_b = alpha * grad_sign_b
    delta_b = delta_b + perturbation_b  # update delta
    delta_b = torch.clip(delta_b, -alpha, alpha)
    delta_b = torch.clip(tgt[:,:,:448,:] + delta_b, 0.0, 1.0) - tgt[:,:,:448,:]

    x[:,:,:448,:] += delta_a
    tgt[:,:,:448,:] += delta_b

    x = x.squeeze(dim=0)
    x = torch.einsum('chw->hwc', x)

    tgt = tgt_adv.squeeze(dim=0)
    tgt = torch.einsum('chw->hwc', tgt)

    return x.detach().numpy(), tgt.detach().numpy()



def construct_adv_AC_fgsm(img, tgt, model, device, alpha=0.031, rand_init=True):

    model.eval()

    x = reshape(img)
    tgt = reshape(tgt)

    if rand_init:
        x_adv = x.detach() + torch.from_numpy(np.random.uniform(-alpha,
                                                                   alpha, x.shape)).float()
        x_adv = torch.clip(x_adv, 0.0, 1.0)
    else:
        x_adv = x.detach()

    x_adv.requires_grad_()

    bool_masked_pos = get_masked_pos(model)
    valid = torch.ones_like(tgt)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    
    bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)   # 变成True/False
    latent = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.to(device))
    pred = model.forward_decoder(latent)
    model.zero_grad()
    with torch.enable_grad():
        loss = model.forward_loss(pred, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))
        
    loss.backward()

    grad_sign = x_adv.grad.detach().sign()
    perturbation = alpha * grad_sign
    x_adv = x_adv + perturbation
    x_adv = torch.min(torch.max(x_adv, x-alpha), x + alpha)
    x_adv = torch.clip(x_adv, 0.0, 1.0)   # L2 bound是指所有像素加起来不能超过budget, 而Linf bound是指每个像素的变化不能超过budget

    x_adv = x_adv.squeeze(dim=0)
    x_adv = torch.einsum('chw->hwc', x_adv)

    tgt = tgt.squeeze(dim=0)
    tgt = torch.einsum('chw->hwc', tgt)

    return x_adv.detach().numpy(), tgt.detach().numpy()



def construct_adv_BC_fgsm(img, tgt, model, device, alpha=0.031, rand_init=True):
    model.eval()

    x = reshape(img)
    tgt = reshape(tgt)

    if rand_init:
        # B图 tgt[:,:,:448,:]
        delta_b = torch.from_numpy(np.random.uniform(-alpha, alpha, tgt[:,:,:448,:].shape)).float()
        delta_b = torch.clip(tgt[:,:,:448,:] + delta_b, 0.0, 1.0) - tgt[:,:,:448,:]

        # C图 x[:,:,448:,:]
        delta_c = torch.from_numpy(np.random.uniform(-alpha, alpha, x[:,:,448:,:].shape)).float()
        delta_c = torch.clip(x[:,:,448:,:] + delta_c, 0.0, 1.0) - x[:,:,448:,:]
    else:
        delta_b = torch.zeros_like(tgt[:,:,:448,:])
        delta_c = torch.zeros_like(x[:,:,448:,:])

    x_adv = x.clone().detach()
    tgt_adv = tgt.clone().detach()

    delta_b.requires_grad_()   # [1,3,448,448]
    delta_c.requires_grad_()   # [1,3,448,448]

    bool_masked_pos = get_masked_pos(model)
    valid = torch.ones_like(tgt)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    tgt_adv[:,:,:448,:] += delta_b  # B图加上噪声
    x_adv[:,:,448:,:] += delta_c  # C图加上噪声
    
    bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)   # 变成True/False
    latent = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt_adv).float().to(device), bool_masked_pos.to(device))
    pred = model.forward_decoder(latent)
    model.zero_grad()
    with torch.enable_grad():
        loss = model.forward_loss(pred, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))
        
    loss.backward()

    grad_sign_b = delta_b.grad.detach().sign()  # [1,3,448,448] 代表方向
    perturbation_b = alpha * grad_sign_b
    delta_b = delta_b + perturbation_b  # update delta
    delta_b = torch.clip(delta_b, -alpha, alpha)
    delta_b = torch.clip(tgt[:,:,:448,:] + delta_b, 0.0, 1.0) - tgt[:,:,:448,:]

    grad_sign_c = delta_c.grad.detach().sign()  # [1,3,448,448] 代表方向
    perturbation_c = alpha * grad_sign_c
    delta_c = delta_c + perturbation_c  # update delta
    delta_c = torch.clip(delta_c, -alpha, alpha)
    delta_c = torch.clip(x[:,:,448:,:] + delta_c, 0.0, 1.0) - x[:,:,448:,:]


    tgt[:,:,:448,:] += delta_b
    x[:,:,448:,:] += delta_c

    x = x.squeeze(dim=0)
    x = torch.einsum('chw->hwc', x)

    tgt = tgt_adv.squeeze(dim=0)
    tgt = torch.einsum('chw->hwc', tgt)

    return x.detach().numpy(), tgt.detach().numpy()



def construct_adv_ABC_fgsm(img, tgt, model, device, alpha=0.031, rand_init=True):

    model.eval()

    x = reshape(img)
    tgt = reshape(tgt)

    if rand_init:
        x_adv = x.detach() + torch.from_numpy(np.random.uniform(-alpha,
                                                                   alpha, x.shape)).float()
        x_adv = torch.clip(x_adv, 0.0, 1.0)

        delta_b = torch.from_numpy(np.random.uniform(-alpha, alpha, tgt[:,:,:448,:].shape)).float()
        delta_b = torch.clip(tgt[:,:,:448,:] + delta_b, 0.0, 1.0) - tgt[:,:,:448,:]
    else:
        x_adv = x.detach()
        delta_b = torch.zeros_like(tgt[:,:,:448,:])

    x_adv.requires_grad_()

    tgt_adv = tgt.clone().detach()
    delta_b.requires_grad_()   # [1,3,448,448]

    bool_masked_pos = get_masked_pos(model)
    valid = torch.ones_like(tgt)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    tgt_adv[:,:,:448,:] += delta_b
    
    bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)   # 变成True/False
    latent = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt_adv).float().to(device), bool_masked_pos.to(device))
    pred = model.forward_decoder(latent)
    model.zero_grad()
    with torch.enable_grad():
        loss = model.forward_loss(pred, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))
        
    loss.backward()

    grad_sign = x_adv.grad.detach().sign()
    perturbation = alpha * grad_sign
    x_adv = x_adv + perturbation
    x_adv = torch.min(torch.max(x_adv, x-alpha), x + alpha)
    x_adv = torch.clip(x_adv, 0.0, 1.0)   # L2 bound是指所有像素加起来不能超过budget, 而Linf bound是指每个像素的变化不能超过budget

    grad_sign_b = delta_b.grad.detach().sign()  # [1,3,448,448] 代表方向
    perturbation_b = alpha * grad_sign_b
    delta_b = delta_b + perturbation_b  # update delta
    delta_b = torch.clip(delta_b, -alpha, alpha)
    delta_b = torch.clip(tgt[:,:,:448,:] + delta_b, 0.0, 1.0) - tgt[:,:,:448,:]

    tgt[:,:,:448,:] += delta_b

    x_adv = x_adv.squeeze(dim=0)
    x_adv = torch.einsum('chw->hwc', x_adv)

    tgt = tgt.squeeze(dim=0)
    tgt = torch.einsum('chw->hwc', tgt)

    return x_adv.detach().numpy(), tgt.detach().numpy()


    
def get_adv_img_adv_tgt_baseline(img, tgt, model_painter, device, attack_id, attack_method, epsilon, num_steps):
    if attack_id == 'attack_A':
        if attack_method == 'FGSM':
            adv_img, adv_tgt = construct_adv_A_fgsm(img, tgt, model_painter, device, alpha=epsilon/255., rand_init=True)
        elif attack_method == 'PGD':
            adv_img, adv_tgt = construct_adv_A_pgd(img, tgt, model_painter, device, epsilon=epsilon/255., num_steps=num_steps, step_size=2/255, rand_init='rand')
    elif attack_id == 'attack_B':
        if attack_method == 'FGSM':
            adv_img, adv_tgt = construct_adv_B_fgsm(img, tgt, model_painter, device, alpha=epsilon/255., rand_init=True)
        elif attack_method == 'PGD':
            adv_img, adv_tgt = construct_adv_B_pgd(img, tgt, model_painter, device, epsilon=epsilon/255., num_steps=num_steps, step_size=2/255, rand_init='rand')
    elif attack_id == 'attack_C':
        if attack_method == 'FGSM':
            adv_img, adv_tgt = construct_adv_C_fgsm(img, tgt, model_painter, device, alpha=epsilon/255., rand_init=True)
        elif attack_method == 'PGD':
            adv_img, adv_tgt = construct_adv_C_pgd(img, tgt, model_painter, device, epsilon=epsilon/255., num_steps=num_steps, step_size=2/255, rand_init='rand')
    elif attack_id == 'attack_AB':
        if attack_method == 'FGSM':
            adv_img, adv_tgt = construct_adv_AB_fgsm(img, tgt, model_painter, device, alpha=epsilon/255., rand_init=True)
        elif attack_method == 'PGD':
            adv_img, adv_tgt = construct_adv_AB_pgd(img, tgt, model_painter, device, epsilon=epsilon/255., num_steps=num_steps, step_size=2/255, rand_init='rand')
    elif attack_id == 'attack_AC':
        if attack_method == 'FGSM':
            adv_img, adv_tgt = construct_adv_AC_fgsm(img, tgt, model_painter, device, alpha=epsilon/255., rand_init=True)
        elif attack_method == 'PGD':
            adv_img, adv_tgt = construct_adv_AC_pgd(img, tgt, model_painter, device, epsilon=epsilon/255., num_steps=num_steps, step_size=2/255, rand_init='rand')
    elif attack_id == 'attack_BC':
        if attack_method == 'FGSM':
            adv_img, adv_tgt = construct_adv_BC_fgsm(img, tgt, model_painter, device, alpha=epsilon/255., rand_init=True)
        elif attack_method == 'PGD':
            adv_img, adv_tgt = construct_adv_BC_pgd(img, tgt, model_painter, device, epsilon=epsilon/255., num_steps=num_steps, step_size=2/255, rand_init='rand')
    elif attack_id == 'attack_ABC':
        if attack_method == 'FGSM':
            adv_img, adv_tgt = construct_adv_ABC_fgsm(img, tgt, model_painter, device, alpha=epsilon/255., rand_init=True)
        elif attack_method == 'PGD':
            adv_img, adv_tgt = construct_adv_ABC_pgd(img, tgt, model_painter, device, epsilon=epsilon/255., num_steps=num_steps, step_size=2/255, rand_init='rand')
    elif attack_id == 'none':
        adv_img, adv_tgt = img, tgt

    return adv_img, adv_tgt