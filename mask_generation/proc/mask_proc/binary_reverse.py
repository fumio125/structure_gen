import cv2
import glob
import numpy as np
import os
from tqdm import tqdm


def bitwise_not_img(b_path, root_tgt_path, text, render_num):
    for i in range(render_num):
        render_dir = f'render_{str(i).zfill(4)}'
        tgt_dir = f'{root_tgt_path}/{render_dir}'
        os.mkdir(tgt_dir)
        img_path_list = sorted(glob.glob(f'../binary/{b_path}/{render_dir}/{text}*.png'))
        for img_path in img_path_list:
            b_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            inverted_b_img = cv2.bitwise_not(b_img)
            _, inverted_b_img = cv2.threshold(inverted_b_img, 200, 255, cv2.THRESH_BINARY)
            cv2.imwrite(f'{tgt_dir}/{os.path.basename(img_path)}', inverted_b_img)


def main(b_dir, text):
    binary_img_list = sorted(os.listdir(b_dir))
    for b_path in tqdm(binary_img_list):
        render_num = len(os.listdir(f'{b_dir}/{b_path}'))
        root_tgt_path = f'../mask_{text}/{b_path}'
        if not os.path.exists(f'../mask_{text}'):
            os.mkdir(f'../mask_{text}')
        os.mkdir(root_tgt_path)
        bitwise_not_img(b_path, root_tgt_path, text, render_num)


if __name__ == '__main__':
    b_dir = '../binary'
    text_list = ['leaf', 'branch']
    for text in text_list:
        main(b_dir=b_dir, text=text)

