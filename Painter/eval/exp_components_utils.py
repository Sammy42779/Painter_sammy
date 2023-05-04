
import random
import numpy as np
from PIL import Image
import glob


FLICKR_LIST = glob.glob('/hhd3/ld/data/flickr30k-images/*.jpg')


NYU_LIST_A = glob.glob('/hhd3/ld/data/nyu_depth_v2/official_splits/test/*/rgb*.jpg')
NYU_LIST_B = glob.glob('/hhd3/ld/data/nyu_depth_v2/official_splits/test/*/sync_depth*.png')
ADE20K_LIST_A = glob.glob('/hhd3/ld/data/ade20k/images/validation/*.jpg')
ADE20K_LIST_B = glob.glob('/hhd3/ld/data/ade20k/annotations_with_color/validation/*.png')
COCO_LIST_A = glob.glob('/hhd3/ld/data/COCO2017/val2017/*.jpg')
COCO_LIST_B = glob.glob('/hhd3/ld/data/COCO2017/pano_sem_seg/panoptic_segm_train2017_with_color/*.png')


### 对于NYU depth任务, 如果tgt没有修改, 那么要对深度图做处理, 使得其范围在0-1之间 
### tgt = np.array(tgt) / 10000.

def load_origin_img(img_path, input_size):
    img2 = Image.open(img_path).convert("RGB")
    img2 = img2.resize((input_size, input_size))
    img2 = np.array(img2) / 255.

    return img2


def load_origin_tgt(tgt_path, input_size, task=None):
    if task == 'nyu_depth':
        tgt2 = Image.open(tgt_path)
        tgt2 = np.array(tgt2) / 10000.
        tgt2 = tgt2 * 255
        tgt2 = Image.fromarray(tgt2).convert("RGB")
        tgt2 = tgt2.resize((input_size, input_size))
        tgt2 = np.array(tgt2) / 255.
    elif task == 'ade20k_segment':
        tgt2 = Image.open(tgt_path)
        tgt2 = tgt2.resize((input_size, input_size))
        tgt2 = np.array(tgt2) / 255. 
    else:
        print('task error')

    return tgt2


def load_other_tgt(tgt_path, input_size):
    ## 不区分任务
    tgt2 = Image.open(tgt_path)
    tgt2 = tgt2.resize((input_size, input_size))
    tgt2 = np.array(tgt2) / 255. 

    return tgt2
 

def get_prompt_gt(img2_path, tgt2_path, input_size, exp_id, transfer_img=None, task=None):

    BLANK_IMG = np.zeros((input_size, input_size, 3), dtype=np.float64)
    WHITE_IMG = np.ones((input_size, input_size, 3), dtype=np.uint8) * 255

    ########## BASELINE ##########  
    if 'baseline' in exp_id: ## baseline
        img2 = load_origin_img(img2_path, input_size)
        tgt2 = load_origin_tgt(tgt2_path, input_size, task=task)

    ########## MASK A ##########    
    elif 'POS_A_mask_A' in exp_id: ## mask_A
        print('@@@@@@ mask A @@@@@@')
        # 将图A替换为空白图像 BLACK or WHITE
        img2 = WHITE_IMG if 'white' in exp_id else BLANK_IMG
        img2 = np.array(img2) / 255.

        tgt2 = load_origin_tgt(tgt2_path, input_size, task=task)

    ########## MASK B ##########
    elif 'POS_B_mask_B' in exp_id: ## mask_B
        print('@@@@@@ mask B @@@@@@')
        # 将图B替换为空白图像 BLACK or WHITE
        img2 = load_origin_img(img2_path, input_size)

        tgt2 = WHITE_IMG if 'white' in exp_id else BLANK_IMG
        tgt2 = np.array(tgt2) / 255.

    ########## MASK A & B ##########
    elif 'POS_AB_mask_AB' in exp_id: ## mask_AB
        # 将图AB替换为空白图像 BLACK or WHITE
        print('@@@@@@ mask AB @@@@@@')
        img2 = WHITE_IMG if 'white' in exp_id else BLANK_IMG
        img2 = np.array(img2) / 255.

        tgt2 = WHITE_IMG if 'white' in exp_id else BLANK_IMG
        tgt2 = np.array(tgt2) / 255.

    ########## EXCHANGE A & B ##########
    elif 'POS_AB_exchange_AB' in exp_id: ## exchange_AB
        print('@@@@@@ exchange AB @@@@@@')
        tgt2 = load_origin_img(img2_path, input_size)
        img2 = load_origin_tgt(tgt2_path, input_size, task=task)

    ########## RANDOM A & SAME TASK ##########
    elif 'POS_A_random_A_same_task' in exp_id: ## random_A_same_task
        print('@@@@@@ random A same task @@@@@@')
        ### same task 需要区分任务
        # 将图A替换为当前任务随机image
        if task == 'nyu_depth':
            img2 = load_origin_img(random.choice(NYU_LIST_A), input_size)
        elif task == 'ade20k_segment':
            img2 = load_origin_img(random.choice(ADE20K_LIST_A), input_size)
        tgt2 = load_origin_tgt(tgt2_path, input_size, task=task)

    ########## RANDOM B & SAME TASK ##########
    elif 'POS_B_random_B_same_task' in exp_id: ## random_B_same_task
        print('@@@@@@ random B same task @@@@@@')
        ### same task 需要区分任务, depth的GT需要特殊处理 /10000.
        # 将图B替换为相同任务随机ground-truth, 不是prompt原始配对的gt
        img2 = load_origin_img(img2_path, input_size)
        
        if task == 'nyu_depth':
            tgt2 = load_origin_tgt(random.choice(NYU_LIST_B), input_size, task=task)
        else:
            tgt2 = load_origin_tgt(random.choice(ADE20K_LIST_B), input_size, task=task)

    ########## RANDOM A & OTHER TASK ##########
    elif 'POS_A_random_A_other_task' in exp_id: ## random_A_other_task
        ### 将图A替换为COCO_A图
        img2 = load_origin_img(random.choice(COCO_LIST_A), input_size)
        tgt2 = load_origin_tgt(tgt2_path, input_size, task=task)

    ########## RANDOM B & OTHER TASK ##########
    elif 'POS_B_random_B_other_task' in exp_id: ## random_B_other_task
        ### 将图B替换为COCO_B图
        img2 = load_origin_img(img2_path, input_size)
        tgt2 = load_other_tgt(random.choice(COCO_LIST_B), input_size)

    ########## ANIMEGAN A ##########
    elif 'POS_A_animeGAN_A' in exp_id: ## animeGAN_A
        # 将图A替换为animeGAN的图像 语义保留, 但是out-of-domain
        img2 = load_origin_img(transfer_img, input_size)
        tgt2 = load_origin_tgt(tgt2_path, input_size, task=task)

    ########## ANIMEGAN B ##########
    elif 'POS_B_animeGAN_B' in exp_id: ## animeGAN_B
        # 将图B替换为animeGAN的图像 语义保留, 但是out-of-domain
        img2 = load_origin_img(img2_path, input_size)
        tgt2 = load_other_tgt(transfer_img, input_size)


    elif exp_id == 'exp_POS_B_random_B_other_task_Flickr':
        # 将图B替换为其他任务随机ground-truth, 不是prompt原始配对的gt
        img2 = load_origin_img(img2_path, input_size)

        ## ground-truth随机 但是是其他数据集
        tgt2 = load_other_tgt(random.choice(FLICKR_LIST), input_size)

    elif exp_id == 'exp_POS_B_random_B_other_task_random':
        # 将图B替换为其他任务随机ground-truth, 不是prompt原始配对的gt
        img2 = load_origin_img(img2_path, input_size)

        ## ground-truth随机 但是是其他数据集
        random_tgt2_path = '/ssd1/ld/ICCV2023/Painter_sammy/test_random.jpg'
        tgt2 = load_other_tgt(random_tgt2_path, input_size)

    return img2, tgt2
