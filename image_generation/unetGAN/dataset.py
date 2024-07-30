import cv2
import glob
import os
from torch.utils import data


class UnetDataset(data.Dataset):
    def __init__(self, source_img_path, input_img_path, resolution, transform=None):
        self.true_img_path = source_img_path
        self.input_img_path = input_img_path
        self.resolution = resolution
        self.transform = transform

    def __getitem__(self, index):
        # print(self.true_img_path[index])
        true_image = cv2.imread(self.true_img_path[index])
        true_image = cv2.cvtColor(true_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.imread(self.input_img_path[index])
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        true_image = self.transform(true_image)
        input_image = self.transform(input_image)

        return true_image, input_image

    def __len__(self):
        return len(self.true_img_path)
