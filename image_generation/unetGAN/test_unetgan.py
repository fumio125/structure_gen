import argparse
import cv2
import glob
import numpy as np
import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm

from dataset import UnetDataset
from unet.unet_model import UNet
from utils.utils import tensor2numpy


def test(args, device):
    train_version = f'{args.species}_{args.train_version}'
    result_dir = f'/data/segmentation/p_{args.species}/train/img'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if args.close:
        CG_dir = 'synthesize_closer'
    elif args.phenotype:
        CG_dir = 'syn_plant'
    elif args.phenobench:
        CG_dir = 'syn_phenobench'
    else:
        CG_dir = 'synthesize'

    if args.real:
        source_img_path = sorted(glob.glob(f'/data/tentative_unet/real_{args.species}/source/*.png'))
        input_img_path = sorted(glob.glob(f'/data/tentative_unet/real_{args.species}/input/*.png'))
    elif args.phenobench:
        source_img_path = sorted(glob.glob(f'/data/tentative_unet/CG_{args.species}/{CG_dir}/leaf_instance/*.png'))
        input_img_path = sorted(glob.glob(f'/data/tentative_unet/CG_{args.species}/{CG_dir}/image/*.png'))
    else:
        source_img_path = sorted(glob.glob(f'/data/tentative_unet/CG_{args.species}/{CG_dir}/source/*.png'))
        input_img_path = sorted(glob.glob(f'/data/tentative_unet/CG_{args.species}/{CG_dir}/input/*.png'))

    # print(source_img_path)
    # print(input_img_path)

    dataset = UnetDataset(
        source_img_path=source_img_path,
        input_img_path=input_img_path,
        resolution=args.test_res,
        transform=test_transform
    )
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print('setup for making dataset finished')

    netG = UNet(n_channels=3, n_classes=3).to(device)

    state_dict = torch.load(f'./experiments/{str(train_version)}/ckpt/model_{str(args.load_e).zfill(3)}.pt')
    netG.load_state_dict(state_dict['state_dict_G'])

    print('Test start!')
    print(len(test_loader))
    netG.eval()
    with torch.no_grad():
        for batch_id, (gt_img, input_img) in enumerate(test_loader):
            input_img = input_img.to(device)
            fake_img = netG(input_img)

            # save the results
            gt_numpy, input_numpy, fake_numpy = tensor2numpy(gt_img.squeeze(0)), tensor2numpy(input_img.squeeze(0)), tensor2numpy(fake_img.squeeze(0))
            if args.species == 'komatsuna':
                for i in range(0, 11):
                    fake_numpy[np.where((fake_numpy[:, :, 0] == i) | (fake_numpy[:, :, 1] == i) | (fake_numpy[:, :, 2] == i))] = (255, 255, 255)
            elif args.species == 'ara' or 'sugar':
                fake_numpy_hsv = cv2.cvtColor(fake_numpy, cv2.COLOR_RGB2HSV)
                lower_blue = np.array([90, 50, 50])
                upper_blue = np.array([120, 255, 255])
                mask = cv2.inRange(fake_numpy_hsv, lower_blue, upper_blue)
                fake_numpy = cv2.inpaint(fake_numpy, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

            cv2.imwrite(os.path.join(result_dir, f'generated_{str(batch_id).zfill(5)}.png'), cv2.cvtColor(fake_numpy, cv2.COLOR_BGR2RGB))
            if (batch_id+1) % 100 == 0:
                print(f'{str(batch_id+1).zfill(5)} image translation finished.')


def main(args):
    device = torch.device(f'cuda:{str(args.gpu_id)}')

    test(args=args, device=device)


def get_args():
    parser = argparse.ArgumentParser(description='Test trained U-Net model')
    parser.add_argument('--load_e', type=int, required=True, help='epochs you want to use for test')
    parser.add_argument('--gpu_id', type=int, required=True, help='specify gpu id you want to use')
    parser.add_argument('--species', type=str, required=True, help='plant species name')
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--close', action='store_true')
    parser.add_argument('--test_res', type=int, default=1024, help='images are resized to this size')
    parser.add_argument('--train_version', type=str, required=True, help='name train dir')
    parser.add_argument('--phenotype', action='store_true')
    parser.add_argument('--phenobench', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)

