import cv2
import glob
import numpy as np
import os
from tqdm import tqdm


def crop_mask(img_path, save_dir):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_size = img.shape[0]

    # separate 0 or 255
    _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    # calculate the coordinate the value of which is 255
    (y, x) = np.where(img % 2 == 1)

    x, y = np.array(x), np.array(y)
    if len(x) == 0 or len(y) == 0:
        return

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    h, w = y_max - y_min, x_max - x_min

    c_x, c_y = int((x_min + x_max) / 2), int((y_min + y_max) / 2)
    crop_size = int(max(h, w) / 2) + 1

    # t:top, b:bottom, l:left, r:right
    t, b, l, r = max(c_y - crop_size, 0), min(c_y + crop_size + 1, img_size), max(c_x - crop_size, 0), min(
        c_x + crop_size + 1, img_size)

    # crop the image and save it with its coordinate and orignal size (npy)
    cropped_img = img[t:b, l:r]
    cv2.imwrite(f'{save_dir}/{os.path.basename(img_path)}', cropped_img)
    bb = np.array([t, b, l, r, cropped_img.shape[0]])
    np.save(f'{save_dir}/{os.path.splitext(os.path.basename(img_path))[0]}.npy', bb)


def main(root_dir, tgt_dir):
    render_dir_list = sorted(os.listdir(root_dir))
    for render_dir in render_dir_list:
        img_dir = os.path.join(root_dir, render_dir)
        save_dir = os.path.join(tgt_dir, render_dir)
        os.mkdir(save_dir)
        img_path_list = sorted(glob.glob(f'{img_dir}/leaf*.png'))
        for img_path in img_path_list:
            crop_mask(img_path, save_dir)


if __name__ == '__main__':
    src_dir = '../mask_leaf'
    dir_name_list = sorted(os.listdir(src_dir))
    for i, dir_name in tqdm(enumerate(dir_name_list)):
        root_dir = os.path.join(src_dir, dir_name)
        tgt_dir = os.path.join('../cropped_mask', dir_name)
        if not os.path.exists('../cropped_mask'):
            os.mkdir('../cropped_mask')
        os.mkdir(tgt_dir)
        main(root_dir, tgt_dir)




