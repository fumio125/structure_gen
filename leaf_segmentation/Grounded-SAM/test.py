import argparse
import cv2
import glob
import numpy as np
import open_clip
import os
import torch
from torch.nn import functional as F
from torchvision import transforms

from evaluation import COCOEvaluator

from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.structures import BitMasks, Instances, Boxes, BoxMode

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict
from huggingface_hub import hf_hub_download

from segment_anything import build_sam, SamPredictor


def load_model_hf(repo_id, filename, ckpt_config_filename, device):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    arg = SLConfig.fromfile(cache_config_file)
    model = build_model(arg)
    arg.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print('Model loaded from {} \n => {}'.format(cache_file, log))
    _ = model.eval()
    return model


def process_grounding_dino(model, img_path, res, text, device):
    # print('processing with grounding dino...')

    box_threshold = 0.28
    text_threshold = 0.25

    transform = transforms.Resize((res, res))
    image_source, image = load_image(img_path)
    image_source, image = cv2.resize(image_source, (res, res)), transform(image)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    return image_source, boxes


def process_segment_anything(model, image_source, bb, device):
    # print('processing with segment anything...')

    model.set_image(image_source)
    H, W, _ = image_source.shape

    # boxes_xyxy = box_ops.box_cxcywh_to_xyxy(bb) * torch.Tensor([W, H, W, H])
    transformed_boxes = model.transform.apply_boxes_torch(bb, image_source.shape[:2]).to(device)

    masks, score, _ = model.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    return masks, score


def eval(img_dir, res_dir, evaluator, device):
    ckpt_repo_id = 'ShilongLiu/GroundingDINO'
    ckpt_filename = 'groundingdino_swinb_cogcoor.pth'
    ckpt_config_filename = 'GroundingDINO_SwinB.cfg.py'
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename, device)

    sam_checkpoint = './ckpt/sam/sam_vit_h_4b8939.pth'
    sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    img_path_list = sorted(glob.glob(f'{img_dir}/*.png'))

    inputs = [
        {'image_id': 1000 + i}
        for i in range(len(img_path_list))
    ]
    outputs = []

    for i, img_path in enumerate(img_path_list):
        img = cv2.imread(img_path)
        v = Visualizer(
            img[:, :, ::-1],
            scale=1,
            instance_mode=ColorMode.SEGMENTATION
        )
        image_source, boxes = process_grounding_dino(model=groundingdino_model, img_path=img_path, res=img.shape[0], text='leaf', device=device)
        pred_boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([img.shape[0], img.shape[1], img.shape[0], img.shape[1]])
        pred_masks, pred_scores = process_segment_anything(model=sam_predictor, image_source=image_source, bb=pred_boxes, device=device)
        # print(pred_scores.shape)
        # print(torch.permute(pred_scores, (1, 0)))
        pred_scores = torch.permute(pred_scores, (1, 0))
        pred_masks = torch.permute(pred_masks, (1, 0, 2, 3))
        mask = {
            "instances": Instances(image_size=(img.shape[0], img.shape[1]), pred_masks=pred_masks[0].to("cpu"))
        }
        out = v.draw_instance_predictions(mask["instances"])
        res_img = out.get_image()[:, :, ::-1]
        cv2.imwrite(f'{res_dir}/{str(i).zfill(5)}.png', res_img)
        pred_boxes = Boxes(pred_boxes)
        pred_masks = pred_masks[0].to("cpu")
        pred_scores = pred_scores[0].to("cpu")
        pred_classes = torch.ones(pred_masks.shape[0])

        instances = Instances(image_size=(img.shape[0], img.shape[1]),
                              pred_boxes=pred_boxes,
                              scores=pred_scores,
                              pred_classes=pred_classes,
                              pred_masks=pred_masks)
        output = {'instances': instances}
        outputs.append(output)

    evaluator.process(inputs, outputs)
    evaluator.evaluate()


def get_args():
    parser = argparse.ArgumentParser(description='test with MaskDINO')
    parser.add_argument('--species', type=str, default='aucuba', help='plant species name')
    parser.add_argument('--gpu_id', type=int, default=0, help='specify gpu id you want to use')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    img_dir = f'/data/segmentation/p_{args.species}/test/img'
    res_dir = f'./results_{args.species}'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    device = torch.device(f'cuda:{str(args.gpu_id)}')
    register_coco_instances('leaf_test', {}, f'/data/segmentation/p_{args.species}/{args.species}_test.json', f'/data/segmentation/p_{args.species}/test')
    evaluator = COCOEvaluator('leaf_test', output_dir=res_dir)
    evaluator.reset()
    eval(img_dir, res_dir, evaluator, device)