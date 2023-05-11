def construct_adv_C_pgd_our(img, tgt, model, device, epsilon, num_steps, step_size, rand_init='rand'):

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
        x_adv.requires_grad_()
        latent_adv = model.forward_encoder(images_normalize(x_adv).float().to(device), images_normalize(tgt).float().to(device), bool_masked_pos.to(device))
        pred_adv = model.forward_decoder(latent_adv)
        latent_adv = torch.cat(latent_adv, dim=-1)
        prob_dist2 = F.softmax(latent_adv, dim=-1)

        # with torch.enable_grad():
        loss1 = model.forward_loss(pred_adv, images_normalize(tgt).float().to(device), bool_masked_pos.to(device), valid.to(device))  ## 1.2999
        kl_loss = F.kl_div(prob_dist1.log(), prob_dist2, reduction='batchmean')  # kl(prob_dist1.log_softmax(dim=-1), prob_dist2.softmax(dim=-1))
        loss = loss1 + kl_loss   # encourage dissimilarity between the two latents
        print(loss1, kl_loss)
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