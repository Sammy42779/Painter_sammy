from constant_utils import *


def construct_adv_C_pgd_breakAC_feat_away_A_C(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand', lam=1, woB=False):

    # print(f'woB:{woB}')

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

    bool_masked_pos = get_masked_pos(model)
    valid = torch.ones_like(tgt)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    
    bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)   # 变成True/False

    if woB == True:
        bool_tgt_all_masked_pos = get_tgt_all_masked_pos(model)
        bool_tgt_all_masked_pos = bool_tgt_all_masked_pos.flatten(1).to(torch.bool)   # 变成True/False

    with torch.no_grad():
        if woB == True:
            clean_a_1, clean_c = model.forward_encoder_ac(images_normalize(x).float().to(device), images_normalize(tgt).float().to(device), bool_tgt_all_masked_pos.to(device))
        else:
            clean_a_1, clean_c = model.forward_encoder_ac(images_normalize(x).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.to(device))

    for i in range(num_steps): 
        model.zero_grad()   # 清空梯度
        x_adv.requires_grad_()
        latent_adv = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.to(device))
        pred_adv = model.forward_decoder(latent_adv)
        latent_adv = torch.cat(latent_adv, dim=-1)

        # basic loss
        loss = model.forward_loss(pred_adv, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))  ## 1.2999
        if woB == False:  ## False 那就是保留B图
            clean_a, adv_c = model.forward_encoder_ac(images_normalize(x_adv).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.to(device))
        elif woB == True:  # True 那就是不保留B图, 输入的时候B图为mask掉的内容
            clean_a, adv_c = model.forward_encoder_ac(images_normalize(x_adv).float().to(device), images_normalize(tgt).float().to(device), bool_tgt_all_masked_pos.to(device))
        loss_ac = L2(clean_c, adv_c)

        if lam in [1,10,100]:  # 1, 0.1, 0.01
            lam = 1 / lam
            loss_ac = loss_ac * lam  
        elif lam in [2,3,4,5]:  # 2,3,4,5
            loss_ac = loss_ac * lam  
        elif lam in [20, 30, 40, 50]:   # 0.2,0.3,0.4,0.5
            lam = lam / 100
            loss_ac = loss_ac * lam  
        elif lam == 10000:
            lam = np.random.beta(1, 1)
            loss_ac = loss_ac * lam  
        elif lam == 10005:
            lam = np.random.beta(0.5, 0.5)
            loss_ac = loss_ac * lam  

        # print(f'loss:{loss}, ac_loss:{loss_ac}, lam:{lam}')

        # print(loss, loss_ac)
        loss = loss + loss_ac
        loss.backward()

        grad_sign = x_adv.grad.detach().sign()
        perturbation = step_size * grad_sign * pos_c
        x_adv = x_adv.detach() + perturbation
        x_adv = torch.min(torch.max(x_adv, x-epsilon), x + epsilon)
        x_adv = torch.clip(x_adv, 0.0, 1.0)   # L2 bound是指所有像素加起来不能超过budget, 而Linf bound是指每个像素的变化不能超过budget

    return reformat_output(x_adv, tgt)



    
def get_adv_img_adv_tgt_breakAC_feat_away_A_C(img, tgt, model_painter, device, attack_id, attack_method, epsilon, num_steps, lam):
    if attack_id == 'attack_C':
        if attack_method == 'PGD':
            adv_img, adv_tgt = construct_adv_C_pgd_breakAC_feat_away_A_C(img, tgt, model_painter, device, 
                                                                epsilon=epsilon/255., num_steps=num_steps, step_size=2/255, rand_init='rand', lam=lam)

    elif attack_id == 'none':
        adv_img, adv_tgt = img, tgt

    return adv_img, adv_tgt