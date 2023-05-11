
from attack_utils_with_clip import *


def get_adv_img_adv_tgt(img, tgt, model_painter, device, attack_id, attack_method, epsilon, step_size):
    if attack_id == 'attack_A':
        if attack_method == 'FGSM':
            adv_img, adv_tgt = construct_adv_A_fgsm(img, tgt, model_painter, device, alpha=epsilon/255., rand_init=True)
    elif attack_id == 'attack_AC':
        if attack_method == 'FGSM':
            adv_img, adv_tgt = construct_adv_AC_fgsm(img, tgt, model_painter, device, alpha=epsilon/255., rand_init=False)
        elif attack_method == 'PGD':
            adv_img, adv_tgt = construct_adv_AC_pgd(img, tgt, model_painter, device, epsilon=epsilon/255., num_steps=step_size, step_size=2/255, rand_init='rand')
    elif attack_id == 'attack_C':
        if attack_method == 'FGSM':
            adv_img, adv_tgt = construct_adv_C_fgsm(img, tgt, model_painter, device, alpha=epsilon/255., rand_init=True)
    elif attack_id == 'attack_AB':
        if attack_method == 'FGSM':
            adv_img, adv_tgt = construct_adv_AB_fgsm(img, tgt, model_painter, device, alpha=epsilon/255., rand_init=True)
    elif attack_id == 'none':
        adv_img, adv_tgt = img, tgt

    return adv_img, adv_tgt