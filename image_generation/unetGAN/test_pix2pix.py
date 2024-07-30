import argparse
import cv2
import glob
import numpy as np
import os
import torch
from torchvision import transforms
import torchvision.transforms.functional as F

from utils.utils import load_yaml
from pix2pix import create_model


def load_pix2pix(device, species, text, condition):
    cfg_path = './pix2pix/configs/pix2pix.yaml' if not condition else './pix2pix/configs/pix2pix_phenotype.yaml'
    cfgs = load_yaml(cfg_path)
    pix2pix_G = create_model(cfgs, device)
    pix2pix_G.setup(cfgs, species, text)
    return pix2pix_G


def translate_mask(mask_path, device, netG, text):
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
    mask_real = F.resize(img=mask_real, size=(256, 256), interpolation=transforms.InterpolationMode.NEAREST, antialias=False)
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

    _, mask_original = cv2.threshold(mask_original, 200, 255, cv2.THRESH_BINARY)

    img_fake = np.transpose(img_fake.cpu().numpy(), (1, 2, 0))
    img_fake = cv2.bitwise_and(img_fake, img_fake, mask=mask_original)

    return img_fake


def main(device, args):
    netG = load_pix2pix(device, species=args.species, text=args.text, condition=args.phenotype)
    mask_path_list = glob.glob(f'{args.src_path}/{args.text}*.png')
    print(mask_path_list)
    root_tgt_dir = f'{args.tgt_path}/{args.species}'
    if not os.path.exists(root_tgt_dir):
        os.mkdir(root_tgt_dir)
        os.mkdir(f'{root_tgt_dir}/leaf')
        os.mkdir(f'{root_tgt_dir}/branch')
    tgt_path = os.path.join(root_tgt_dir, args.text)
    num = len(glob.glob(f'{tgt_path}/*.png'))
    for i, mask_path in enumerate(mask_path_list):
        img = translate_mask(mask_path, device, netG, args.text)
        cv2.imwrite(f'{tgt_path}/{str(i+num).zfill(6)}.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def get_args():
    parser = argparse.ArgumentParser(description='making dataset for synthesize experiment')
    parser.add_argument('--gpu_id', type=int, required=True, help='specify gpu id you want to use')
    parser.add_argument('--src_path', type=str, default='/data/src/synthesize_plant/mask_branch/0002700/render_0000')
    parser.add_argument('--tgt_path', type=str, default='./tmp', help='path to dataset using in the experiment')
    parser.add_argument('--text', type=str, required=True, help='text prompt')
    parser.add_argument('--species', type=str, required=True, help='plant species name')
    parser.add_argument('--phenotype', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    device = torch.device(f'cuda:{str(args.gpu_id)}')
    main(device, args)



