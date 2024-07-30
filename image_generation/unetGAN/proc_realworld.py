import argparse
import cv2
import glob
import os
import random
import torch
from tqdm import tqdm

from utils.preproc import create_organ_replace_img, create_thesis_data
from utils.utils import *


def main(args, mode):
    if args.type == 'proc':
        src_dir = f'/data/src/real_plant/{args.species}/unet/{mode}'
    else:
        src_dir = f'/data/src/real_plant/{args.species}/thesis'
    tgt_dir = f'/data/tentative_unet/real_{args.species}/{mode}'
    device = torch.device(f'cuda:{str(args.gpu_id)}')
    img_path = sorted(glob.glob(os.path.join(src_dir, '*.png')))
    if not os.path.exists(tgt_dir):
        os.mkdir(tgt_dir)
        os.mkdir(os.path.join(tgt_dir, 'source'))
        os.mkdir(os.path.join(tgt_dir, 'input'))
    m1, m2, m3 = setup_pretrained_models(device=device, species=args.species, condition=args.phenotype)
    if args.type == 'proc':
        with tqdm(range(0, len(img_path))) as pbar:
            for i in pbar:
                source_img, input_img = create_organ_replace_img(model1=m1, model2=m2, model3=m3[0],
                                                                 img_path=img_path[i], res=args.res, text='leaf',
                                                                 device=device)
                if args.species == 'ara':
                    input_img_90 = cv2.rotate(input_img, cv2.ROTATE_90_CLOCKWISE)
                    input_img_180 = cv2.rotate(input_img, cv2.ROTATE_180)
                    input_img_270 = cv2.rotate(input_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    source_img_90 = cv2.rotate(source_img, cv2.ROTATE_90_CLOCKWISE)
                    source_img_180 = cv2.rotate(source_img, cv2.ROTATE_180)
                    source_img_270 = cv2.rotate(source_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    cv2.imwrite(f'{tgt_dir}/source/{str(4*i).zfill(6)}.png', source_img[:,:,::-1])
                    cv2.imwrite(f'{tgt_dir}/source/{str(4*i+1).zfill(6)}.png', source_img_90[:,:,::-1])
                    cv2.imwrite(f'{tgt_dir}/source/{str(4*i+2).zfill(6)}.png', source_img_180[:,:,::-1])
                    cv2.imwrite(f'{tgt_dir}/source/{str(4*i+3).zfill(6)}.png', source_img_270[:,:,::-1])
                    cv2.imwrite(f'{tgt_dir}/input/{str(4*i).zfill(6)}.png', input_img[:,:,::-1])
                    cv2.imwrite(f'{tgt_dir}/input/{str(4*i+1).zfill(6)}.png', input_img_90[:,:,::-1])
                    cv2.imwrite(f'{tgt_dir}/input/{str(4*i+2).zfill(6)}.png', input_img_180[:,:,::-1])
                    cv2.imwrite(f'{tgt_dir}/input/{str(4*i+3).zfill(6)}.png', input_img_270[:,:,::-1])
                else:
                    cv2.imwrite(f'{tgt_dir}/source/{str(i).zfill(6)}.png', cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB))
                    cv2.imwrite(f'{tgt_dir}/input/{str(i).zfill(6)}.png', cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

        if not args.phenotype:
            new_img_path = glob.glob(os.path.join(tgt_dir, 'input', '*.png'))
            with tqdm(range(0, len(img_path))) as pbar:
                for i in pbar:
                    source_img, input_img = create_organ_replace_img(model1=m1, model2=m2, model3=m3[1],
                                                     img_path=new_img_path[i], res=args.res, text='branch',
                                                     device=device)
                    cv2.imwrite(f'{tgt_dir}/input/{str(i).zfill(6)}.png', cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        print(f'{mode} dataset finished making -> total data: {len(img_path)}')
    else:
        with tqdm(range(0, len(img_path))) as pbar:
            for i in pbar:
                create_thesis_data(model1=m1, model2=m2, model3=m3[0], img_path=img_path[i], res=args.res, text='leaf', device=device)


def get_args():
    parser = argparse.ArgumentParser(description='making dataset for real world experiment')
    parser.add_argument('--gpu_id', type=int, required=True, help='specify gpu id you want to use')
    parser.add_argument('--res', type=int, default=1024, help='images are resized to this size')
    parser.add_argument('--species', type=str, required=True, help='plant species name')
    parser.add_argument('--phenotype', action='store_true')
    parser.add_argument('--type', type=str, default='proc', help='proc or thesis')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    # mode: train or test
    mode_list = ['train', 'test']
    for mode in mode_list:
        main(args, mode)




