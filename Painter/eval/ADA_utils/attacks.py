from constant_utils import *

def get_pos_matrix(x, tgt, exp_pos):

    pos_AC = torch.zeros_like(x)
    pos_BD = torch.zeros_like(tgt)

    if exp_pos == 'attack_A':
        pos_AC[:,:,:448,:] = 1  # pos_a
    elif exp_pos == 'attack_B':
        pos_BD[:,:,:448,:] = 1  # pos_b
    elif exp_pos == 'attack_C':
        pos_AC[:,:,448:,:] = 1  # pos_c
    elif exp_pos == 'attack_AB':
        pos_AC[:,:,:448,:] = 1  # pos_a
        pos_BD[:,:,:448,:] = 1  # pos_b
    elif exp_pos == 'attack_AC':
        pos_AC[:,:,:,:] = 1  # pos_ac
    elif exp_pos == 'attack_BC':
        pos_BD[:,:,:448,:] = 1  # pos_b
        pos_AC[:,:,448:,:] = 1  # pos_c
    elif exp_pos == 'attack_ABC':
        pos_AC[:,:,:,:] = 1  # pos_ac
        pos_BD[:,:,:448,:] = 1  # pos_b

    return pos_AC, pos_BD



def construct_adv_PGD(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand',
                        exp_method='', exp_pos='',
                        break_AC=False, lam_AC=0.01, with_B=False, 
                        mask_B=False, ignore_D_loss=False, lam_AB=0.1, mask_ratio=0.75):
    """
    exp_method: 'VA', 'AA', 'DA'. This refers to how to generate adversarial data.
    exp_pos: 'attack_A', 'attack_B', 'attack_C', 'attack_AB', 'attack_AC', 'attack_BC', 'attack_ABC'. This refers to which part of the visual prompt is perturbed.
    break_AC: Corresponds to 'DA', which is distribution attack.
    mask_B: Corresponds to 'AA', which is alignment attack.
    """

    model.eval()

    x = reshape(img)
    tgt = reshape(tgt)

    pos_AC, pos_BD = get_pos_matrix(x, tgt, exp_pos)

    if rand_init:
        x_adv = x.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, x.shape)).float() * pos_AC
        x_adv = torch.clip(x_adv, 0.0, 1.0)

        tgt_adv = tgt.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, tgt.shape)).float() * pos_BD
        tgt_adv = torch.clip(tgt_adv, 0.0, 1.0)
    else:
        x_adv = x.detach()
        tgt_adv = tgt.detach()

    bool_masked_pos_basic = get_masked_pos(model, mask_B=False, mask_ratio=mask_ratio) # wt
    bool_masked_pos_basic = bool_masked_pos_basic.flatten(1).to(torch.bool)   # 变成True/False

    if mask_B:  # alignment attack 
        bool_masked_mask_B = get_masked_pos(model, mask_B=mask_B, mask_ratio=mask_ratio) # wt
        bool_masked_mask_B = bool_masked_mask_B.flatten(1).to(torch.bool)   # 变成True/False

    valid = torch.ones_like(tgt)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    for _ in range(num_steps): 
        x_adv.requires_grad_()   
        tgt_adv.requires_grad_()    

        ### Forward pass
        latent_adv = model.forward_encoder(images_normalize(x_adv).float().to(device), 
                                           images_normalize(tgt_adv).float().to(device), 
                                           bool_masked_pos_basic.to(device))
        pred_adv = model.forward_decoder(latent_adv)


        ### Calculate the loss
        # L_pred(y', GT)
        loss = model.forward_loss(pred_adv, images_normalize(tgt).float().to(device), bool_masked_pos_basic.to(device), valid.to(device))  

        ## Distribution Attack (DA)
        if break_AC:
            latent_adv_tmp = torch.cat(latent_adv, dim=-1)
            latent_adv_a, latent_adv_c = latent_adv_tmp[:,:28,:,:], latent_adv_tmp[:,28:,:,:]  # distangle latent_adv_a and latent_adv_c
            loss_dist = lam_AC * L2(latent_adv_a, latent_adv_c)   # L_dist

            loss = loss + loss_dist
        
        # Alignment Attack (AA)
        if mask_B:
            # 把B mask后得到的变量
            mask_B_latent = model.forward_encoder(images_normalize(x_adv).float().to(device), 
                                                  images_normalize(tgt_adv).float().to(device), 
                                                  bool_masked_mask_B.to(device))   # mask B
            mask_B_pred = model.forward_decoder(mask_B_latent)
            # mask loss
            loss_mask = model.forward_loss(mask_B_pred, images_normalize(tgt).float().to(device), 
                                           bool_masked_mask_B.to(device), valid.to(device), ignore_D_loss=ignore_D_loss)  # wt
            loss_align = lam_AB * loss_mask  # L_align

            loss = loss + loss_align

        ### Zero all existing gradients
        model.zero_grad()
        
        ### Backward pass
        loss.backward()

        if exp_pos in ['attack_A', 'attack_C', 'attack_AB', 'attack_AC', 'attack_BC', 'attack_ABC']:
            ### Collect gradients of inputs
            grad_sign_x = x_adv.grad.detach().sign()
            perturbation_x = step_size * grad_sign_x * pos_AC
            x_adv = x_adv.detach() + perturbation_x   # 已经没有梯度了, 不需要后期再对对抗样本梯度清零
            ### Clip the perturbations to epsilon
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)  # [检查] 对抗样本的范围是tgt还是x
            x_adv = torch.clip(x_adv, 0.0, 1.0)   # L2 bound是指所有像素加起来不能超过budget, 而Linf bound是指每个像素的变化不能超过budget
        if exp_pos in ['attack_B', 'attack_AB', 'attack_BC', 'attack_ABC']:
            grad_sign_tgt = tgt_adv.grad.detach().sign()
            perturbation_tgt = step_size * grad_sign_tgt * pos_BD
            tgt_adv = tgt_adv.detach() + perturbation_tgt
            tgt_adv = torch.min(torch.max(tgt_adv, tgt - epsilon), tgt + epsilon)  # [检查] 对抗样本的范围是tgt还是x
            tgt_adv = torch.clip(tgt_adv, 0.0, 1.0)   # L2 bound是指所有像素加起来不能超过budget, 而Linf bound是指每个像素的变化不能超过budget

    return reformat_output(x_adv, tgt_adv)
