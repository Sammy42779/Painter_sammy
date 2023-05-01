
import random
import numpy as np
from PIL import Image
import glob


FLICKR_LIST = glob.glob('/hhd3/ld/data/flickr30k-images/*.jpg')
ADE20K_LIST = glob.glob('/hhd3/ld/data/ade20k/annotations_with_color/validation/*.png')
COCO_LIST = glob.glob('/hhd3/ld/data/COCO2017/pano_sem_seg/panoptic_segm_train2017_with_color/*.png')


def get_prompt_gt(img2_path, tgt2_path, input_size, exp_id, transfer_img=None):

    BLANK_IMG = np.zeros((input_size, input_size, 3), dtype=np.float64)
    WHITE_IMG = np.ones((input_size, input_size, 3), dtype=np.uint8) * 255

    if exp_id == 'exp_mapping_1_1_b' or exp_id == 'exp_mapping_1_1_b_val':
        # 1-1(b): 将图B替换为ADE20K segmentation任务随机ground-truth, 不是prompt原始配对的gt
        # load the shared prompt image pair
        ## prompt不变
        img2 = Image.open(img2_path).convert("RGB")
        img2 = img2.resize((input_size, input_size))
        img2 = np.array(img2) / 255.

        ## ground-truth随机
        # random_tgt2_path = '/hhd3/ld/data/ade20k/annotations_with_color/training/ADE_train_00014844.png'
        # random_tgt2_path = '/hhd3/ld/data/ade20k/annotations_with_color/validation/ADE_val_00000038.png'
        random_tgt2_path = random.choice(ADE20K_LIST)
        tgt2 = Image.open(random_tgt2_path)
        tgt2 = tgt2.resize((input_size, input_size))
        tgt2 = np.array(tgt2) / 255.

    elif exp_id == 'exp_mapping_1_1_a':
        # 1-1(b): 将图B替换为其他任务随机ground-truth, 不是prompt原始配对的gt
        # load the shared prompt image pair
        ## prompt不变
        img2 = Image.open(img2_path).convert("RGB")
        img2 = img2.resize((input_size, input_size))
        img2 = np.array(img2) / 255.

        ## ground-truth随机 但是是其他任务
        # random_tgt2_path = '/hhd3/ld/data/COCO2017/pano_sem_seg/panoptic_segm_train2017_with_color/000000391460.png'
        random_tgt2_path = random.choice(COCO_LIST)
        tgt2 = Image.open(random_tgt2_path)
        tgt2 = tgt2.resize((input_size, input_size))
        tgt2 = np.array(tgt2) / 255.

    elif exp_id == 'exp_input_distribution_2_1_a':
        # 2-1(a): 将图A替换成train OOD图像, 替换input distribution
        # 替换img2_path为Flickr30K里的图像
        ## prompt改变OOD
        ood_img2_path = random.choice(FLICKR_LIST)
        img2 = Image.open(ood_img2_path).convert("RGB")
        img2 = img2.resize((input_size, input_size))
        img2 = np.array(img2) / 255.

        # ground-truth不变
        tgt2 = Image.open(tgt2_path)
        tgt2 = tgt2.resize((input_size, input_size))
        tgt2 = np.array(tgt2) / 255.

    elif exp_id == 'exp_output_distribution_3_1_a':
        # 3-1(a): 将图B替换成train OOD图像, 替换output distribution
        # 替换tgt2_path为Flickr30K里的图像
        ## prompt不变
        img2 = Image.open(img2_path).convert("RGB")
        img2 = img2.resize((input_size, input_size))
        img2 = np.array(img2) / 255.

        # ground-truth改变OOD
        ood_tgt2_path = random.choice(FLICKR_LIST)
        tgt2 = Image.open(ood_tgt2_path)
        tgt2 = tgt2.resize((input_size, input_size))
        tgt2 = np.array(tgt2) / 255.

    elif exp_id == 'exp_format_4_1_a':
        # 4-1(a): 没有format, 没有prompt, 因此img2和tgt2都是空白图像
        # img2 = Image.open(img2_path).convert("RGB")
        # img2 = img2.resize((input_size, input_size))
        img2 = BLANK_IMG
        img2 = np.array(img2) / 255.

        # tgt2 = Image.open(tgt2_path)
        # tgt2 = tgt2.resize((input_size, input_size))
        tgt2 = BLANK_IMG
        tgt2 = np.array(tgt2) / 255.

    elif 'baseline' in exp_id: ## baseline
        # load the shared prompt image pair
        img2 = Image.open(img2_path).convert("RGB")
        img2 = img2.resize((input_size, input_size))
        img2 = np.array(img2) / 255.

        tgt2 = Image.open(tgt2_path)
        tgt2 = tgt2.resize((input_size, input_size))
        tgt2 = np.array(tgt2) / 255.

    ### 
    elif exp_id in ['exp_POS_A_mask_A', 'exp_POS_A_mask_A_white']:
        print('%%%%%%%%%%%%%%%%%%exp_POS_A_mask_A')
        # 将图A替换为空白图像
        img2 = WHITE_IMG if 'white' in exp_id else BLANK_IMG
        img2 = np.array(img2) / 255.

        tgt2 = Image.open(tgt2_path)
        tgt2 = tgt2.resize((input_size, input_size))
        tgt2 = np.array(tgt2) / 255.

    ### 
    elif exp_id in ['exp_POS_B_mask_B', 'exp_POS_B_mask_B_white']:
        print('%%%%%%%%%%%%%%%%%%exp_POS_B_mask_B')
        # 将图B替换为空白图像
        img2 = Image.open(img2_path).convert("RGB")
        img2 = img2.resize((input_size, input_size))
        img2 = np.array(img2) / 255.

        tgt2 = WHITE_IMG if 'white' in exp_id else BLANK_IMG
        tgt2 = np.array(tgt2) / 255.

    ### 
    elif exp_id in ['exp_POS_AB_mask_AB', 'exp_POS_AB_mask_AB_white']:
        print('%%%%%%%%%%%%%%%%%%exp_POS_AB_mask_AB')
        # 将图AB替换为空白图像
        img2 = WHITE_IMG if 'white' in exp_id else BLANK_IMG
        img2 = np.array(img2) / 255.

        tgt2 = WHITE_IMG if 'white' in exp_id else BLANK_IMG
        tgt2 = np.array(tgt2) / 255.

    elif exp_id == 'exp_POS_A_ood_A_animeGAN': ## 
        # 将图A替换为animeGAN的图像 语义保留, 但是out-of-domain
        # transfer_img = "/hhd3/ld/data/ade20k/AnimeGANv2/training/ADE_train_00009574_animeGAN.png"
        img2 = Image.open(transfer_img).convert("RGB")
        img2 = img2.resize((input_size, input_size))
        img2 = np.array(img2) / 255.

        tgt2 = Image.open(tgt2_path)
        tgt2 = tgt2.resize((input_size, input_size))
        tgt2 = np.array(tgt2) / 255.

    elif exp_id == 'exp_POS_B_ood_B_animeGAN': ## 
        img2 = Image.open(img2_path).convert("RGB")
        img2 = img2.resize((input_size, input_size))
        img2 = np.array(img2) / 255.

        # 将图B替换为animeGAN的图像 语义保留, 但是out-of-domain
        # transfer_img = "/hhd3/ld/data/ade20k/AnimeGANv2/validation/ADE_train_00009574_animeGAN.png"
        tgt2 = Image.open(transfer_img)
        tgt2 = tgt2.resize((input_size, input_size))
        tgt2 = np.array(tgt2) / 255.


    return img2, tgt2
