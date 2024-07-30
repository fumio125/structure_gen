import argparse
import cv2
import glob
import numpy as np
import os
import scipy import ndimage
import time
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import Instances
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer

from evaluation import COCOEvaluator
from maskdino import add_maskdino_config


def extract_largest_mask(mask, device):
    mask = mask.to('cpu').detach().numpy().copy()
    labeled_mask, num_features = ndimage.label(mask)

    if num_features == 0:
        mask = torch.from_numpy(mask).float().to(device)
        return mask

    label_sizes = np.bincount(labeled_mask.ravel())
    label_sizes[0] = 0

    largest_label = label_sizes.argmax()
    largest_mask = labeled_mask == largest_label

    largest_mask = torch.from_numpy(largest_mask).float().to(device)

    return largest_mask


def remove_small_masks(output, img, thr, device):
    masks = output["instances"].pred_masks
    boxes = output["instances"].pred_boxes
    scores = output["instances"].scores
    pred_classes = output["instances"].pred_classes
    kept_masks = torch.zeros([0, img.shape[0], img.shape[1]]).to(device)
    kept_boxes = torch.zeros([0, 4]).to(device)
    kept_scores = torch.zeros(0).to(device)
    kept_classes = torch.zeros(0).to(device)
    for i in range(len(masks)):
        mask = masks[i]
        mask = extract_largest_mask(mask, device)
        area = torch.sum(mask).item()
        if area > thr:
            kept_masks = torch.cat((kept_masks, mask.unsqueeze(0)), dim=0)
            kept_boxes = torch.cat((kept_boxes, boxes[i].tensor.to(device)), dim=0)
            kept_scores = torch.cat((kept_scores, scores[i].unsqueeze(0)), dim=0)
            kept_classes = torch.cat((kept_classes, pred_classes[i].unsqueeze(0)), dim=0)
    new_output = {
        "instances": Instances(image_size=(img.shape[0], img.shape[1]), pred_masks=kept_masks,
                                   pred_boxes=Boxes(kept_boxes), scores=kept_scores, pred_classes=kept_classes)
    }
    return new_output


def test(img_dir, res_dir, evaluator):
    predictor = DefaultPredictor(cfg)

    img_path_list = sorted(glob.glob(f'{img_dir}/*.png'))

    inputs = [
        {'image_id': 1000 + i}
        for i in range(len(img_path_list))
    ]
    outputs = []

    for i, img_path in enumerate(img_path_list):
        img = cv2.imread(img_path)
        output_tmp = predictor(img)
        output = remove_small_masks(output_tmp, img, cfg.thr, cfg.CUDA)
        outputs.append(output)
        v = Visualizer(
            img[:, :, ::-1],
            scale=1,
            instance_mode=ColorMode.SEGMENTATION
        )
        mask = {
            "instances": Instances(image_size=(img.shape[0], img.shape[1]), pred_masks=output["instances"].pred_masks.to("cpu"))
        }
        out = v.draw_instance_predictions(mask["instances"])
        res_img = out.get_image()[:, :, ::-1]
        cv2.imwrite(f'{res_dir}/{str(i).zfill(5)}.png', res_img)

    evaluator.process(inputs, outputs)
    evaluator.evaluate()


def setup(args):
    register_coco_instances('leaf_train', {}, f'/data/segmentation/p_{args.species}/{args.species}_train.json', f'/data/segmentation/p_{args.species}/train')
    register_coco_instances('leaf_test', {}, f'/data/segmentation/p_{args.species}/{args.species}_test.json', f'/data/segmentation/p_{args.species}/test')
    cfg = get_cfg()

    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.TRAIN = ('leaf_train',)
    cfg.DATASETS.TEST = ('leaf_test',)
    cfg.OUTPUT_DIR = f'./results/results_{args.species}'
    cfg.CUDA = f'cuda:{args.gpu_id}'
    cfg.thr = args.threshold
    cfg.freeze()

    return cfg


def get_args():
    parser = argparse.ArgumentParser(description='test with MaskDINO')
    parser.add_argument('--species', type=str, default='aucuba', help='plant species name')
    parser.add_argument('--config_file', default='', help='path to config file')
    parser.add_argument('--gpu_id', type=int, default=0, help='specify gpu id you want to use')
    parser.add_argument('--threshold', type=int, default=0, help='threshold of area')
    parser.add_argument('--actual', action='store_true')
    parser.add_argument('--scatter', action='store_true')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    cfg = setup(args)
    img_dir = f'/data/segmentation/p_{args.species}/test/img'
    if args.actual or args.scatter:
        res_dir = f'./results_{args.species}_comp'
    else:
        res_dir = f'./results/results_{args.species}'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], output_dir=res_dir)
    evaluator.reset()
    test(img_dir, res_dir, evaluator)
