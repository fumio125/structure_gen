import argparse
import cv2
import glob
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
import os
import random

from utils.utils import load_yaml
from pix2pix import create_model


def load_pix2pix(device, species, text, condition):
    cfg_path = './pix2pix/configs/pix2pix.yaml' if not condition else './pix2pix/configs/pix2pix_phenotype.yaml'
    cfgs = load_yaml(cfg_path)
    pix2pix_G = create_model(cfgs, device)
    pix2pix_G.setup(cfgs, species, text)
    return pix2pix_G


def process_img(mask_path, device, processed_img, netG, text, condition):
    pix2pix_size = 256 if not condition else 128
    mask_original = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if text == 'leaf':
        # set bounding box => (top, bottom, left, right, bb_size)
        bb = np.load(os.path.join(os.path.dirname(mask_path), os.path.splitext(os.path.basename(mask_path))[0]+'.npy'), allow_pickle=True)
        t, b, l, r = bb[0], bb[1], bb[2], bb[3]
        w, h = b - t, r - l

        if w > h:
            gap = w - h
            if l == 0:
                mask_real = np.pad(mask_original, ((0, 0), (0, gap)))
            else:
                mask_real = np.pad(mask_original, ((0, 0), (gap, 0)))
        elif h > w:
            gap = h - w
            if t == 0:
                mask_real = np.pad(mask_original, ((0, gap), (0, 0)))
            else:
                mask_real = np.pad(mask_original, ((gap, 0), (0, 0)))
        else:
            mask_real = mask_original

    else:
        mask_real = mask_original
        t, b, l, r = 0, mask_original.shape[0], 0, mask_original.shape[0]

    mask_size = mask_real.shape[0]
    mask_real = transforms.ToTensor()(mask_real)

    # resize mask to the pix2pix input size (=256) and transform
    mask_real = F.resize(img=mask_real, size=(pix2pix_size, pix2pix_size), interpolation=transforms.InterpolationMode.NEAREST, antialias=False)
    mask_real = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(mask_real.to(torch.float32).squeeze(0).repeat(3, 1, 1))
    mask_real = mask_real.to(device)

    # generate fake image
    netG.set_input(mask_real.unsqueeze(0))
    netG.test()
    img_fake = netG.get_current_visuals()['fake_B'][0]

    # paste img_fake on processed_img after processing
    img_fake = (img_fake + 1) / 2.0 * 255.0

    if text == 'leaf':
        if w > h:
            if l == 0:
                img_fake = F.resize(img_fake, size=(mask_size, mask_size))[:, :, :mask_size-gap]
            else:
                img_fake = F.resize(img_fake, size=(mask_size, mask_size))[:, :, gap:]
        elif h > w:
            if t == 0:
                img_fake = F.resize(img_fake, size=(mask_size, mask_size))[:, :mask_size-gap, :]
            else:
                img_fake = F.resize(img_fake, size=(mask_size, mask_size))[:, gap:, :]
        else:
            img_fake = F.resize(img_fake, size=(mask_size, mask_size))
    else:
        img_fake = F.resize(img_fake, size=(mask_size, mask_size))

    mask_original = torch.tensor(mask_original).to(device).ge(200)
    img_fake = torch.masked_fill(input=img_fake, mask=~mask_original, value=0)

    img1 = torch.masked_fill(input=torch.permute(processed_img[t:b, l:r], (2, 0, 1)), mask=mask_original, value=0)
    img2 = torch.bitwise_or(img1, img_fake.to(torch.uint8)[[2, 1, 0], :, :])

    processed_img[t:b, l:r] = torch.permute(img2, (1, 2, 0))

    return processed_img


def main(args, netG, device, src_img_path, src_branch_mask_path, src_leaf_mask_path, tgt_path, id):
    source_img = cv2.imread(f'{src_img_path}')
    lower_white = (0, 0, 200)
    upper_white = (180, 30, 255)
    hsv_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2HSV)
    mask_img = cv2.inRange(hsv_img, lower_white, upper_white)
    mask_img = cv2.bitwise_not(mask_img)
    processed_img = torch.tensor(source_img).to(device)

    l_mask_path_list = glob.glob(f'{src_leaf_mask_path}/leaf*.png')
    if not args.phenotype:
        b_mask_path_list = glob.glob(f'{src_branch_mask_path}/branch*.png')
        for mask_path in b_mask_path_list:
            processed_img = process_img(mask_path, device, processed_img, netG[1], text='branch', condition=args.phenotype)
    for mask_path in l_mask_path_list:
        processed_img = process_img(mask_path, device, processed_img, netG[0], text='leaf', condition=args.phenotype)

    if args.amodal:
        bg_id = 0
        bg_img = cv2.imread(f'/data/src/background/amodal_cropped/{str(bg_id).zfill(5)}.png')
    elif args.species == 'komatsuna':
        bg_id = random.randint(0, 4)
        bg_img = cv2.imread(f'/data/src/background/CG_plant/komatsuna/{str(bg_id).zfill(6)}.png')
    elif args.species == 'ara':
        bg_id = random.randint(0, 13)
        bg_img = cv2.imread(f'/data/src/background/phenotype/new_background/{str(bg_id).zfill(5)}.png')
    elif args.species == 'sugar':
        bg_img = np.ones((512, 512, 3)) * 255
    else:
        bg_id = random.randint(0, 5)
        bg_img = cv2.imread(f'/data/src/background/cropped/{str(bg_id).zfill(5)}.png')
    if args.img_res != processed_img.shape[0]:
        mask_img = F.resize(img=torch.tensor(mask_img).unsqueeze(0), size=(args.img_res, args.img_res), interpolation=transforms.InterpolationMode.NEAREST, antialias=False)
        mask_img = mask_img.squeeze(0).numpy()
        processed_img = F.resize(img=torch.permute(processed_img, (2, 0, 1)).unsqueeze(0), size=(args.img_res, args.img_res), interpolation=transforms.InterpolationMode.NEAREST, antialias=False)
        processed_img = torch.permute(processed_img.squeeze(0), (1, 2, 0))
        bg_img = cv2.resize(bg_img, (args.img_res, args.img_res))
    processed_img = processed_img.cpu().numpy()

    input_img = np.where(mask_img[:args.img_res, :args.img_res, np.newaxis] == 0, bg_img, processed_img)
    if args.amodal:
        cv2.imwrite(f'{tgt_path}/{str(id).zfill(6)}.png', input_img)
    else:
        cv2.imwrite(f'{tgt_path}/input/{str(id).zfill(6)}.png', input_img)
        cv2.imwrite(f'{tgt_path}/source/{str(id).zfill(6)}.png', source_img)


def get_args():
    parser = argparse.ArgumentParser(description='making dataset for synthesize experiment')
    parser.add_argument('--gpu_id', type=int, required=True, help='specify gpu id you want to use')
    parser.add_argument('--species', type=str, required=True, help='plant species name')
    parser.add_argument('--close', action='store_true')
    parser.add_argument('--render_num', type=int, default=8, help='render times')
    parser.add_argument('--phenotype', action='store_true')
    parser.add_argument('--img_res', type=int, default=1024, help='resolution of output image')
    parser.add_argument('--amodal', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    CG_dir = 'syn_plant'

    if args.amodal:
        src_img_dir = f'/data/amodal_segmentation/data_latest/source_img'
        src_leaf_mask_dir = f'/data/amodal_segmentation/data_latest/cropped_mask'
        src_branch_mask_dir = f'/data/amodal_segmentation/data_latest/mask_branch'
        tgt_path = f'/data/amodal_segmentation/data_latest/textured_img'
    else:
        root_dir = f'/data/src/{CG_dir}/{args.species}'
        src_img_dir = f'{root_dir}/img'
        src_leaf_mask_dir = f'{root_dir}/cropped_mask'
        src_branch_mask_dir = f'{root_dir}/mask_branch' if not args.phenotype else 'none'
        tgt_path = f'/data/tentative_unet/CG_{args.species}/{CG_dir}'

    src_img_path_list = sorted(glob.glob(f'{src_img_dir}/*.png'))
    if not os.path.exists(tgt_path):
        os.mkdir(tgt_path)
        if not args.amodal:
            os.mkdir(os.path.join(tgt_path, 'input'))
            os.mkdir(os.path.join(tgt_path, 'source'))

    device = torch.device(f'cuda:{str(args.gpu_id)}')
    netG = []
    netG.append(load_pix2pix(device, species=args.species, text='leaf', condition=args.phenotype))
    if not args.phenotype:
        netG.append(load_pix2pix(device, species=args.species, text='branch', condition=args.phenotype))

    plant_dir_list = sorted(os.listdir(src_leaf_mask_dir))
    for i, src_img_path in tqdm(enumerate(src_img_path_list)):
        plant_dir_id = int(i / args.render_num)
        src_branch_mask_path = os.path.join(src_branch_mask_dir, plant_dir_list[plant_dir_id], f'render_{str(i%args.render_num).zfill(4)}') if not args.phenotype else []
        src_leaf_mask_path = os.path.join(src_leaf_mask_dir, plant_dir_list[plant_dir_id], f'render_{str(i%args.render_num).zfill(4)}')

        main(args, netG, device, src_img_path, src_branch_mask_path, src_leaf_mask_path, tgt_path, i)








