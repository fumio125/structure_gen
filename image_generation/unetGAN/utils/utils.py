import numpy as np
import os
import torch
import yaml

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
# from mobile_sam import sam_model_registry, SamPredictor
from segment_anything import build_sam, SamPredictor
from huggingface_hub import hf_hub_download

from pix2pix import create_model


def tensor2numpy(tensor):
    ndarray = ((np.transpose(tensor.detach().cpu().numpy(), (1, 2, 0)) + 1) / 2.0 * 255.0)
    # ndarray = np.transpose(tensor.detech().cpu().numpy(), (1, 2, 0))
    return ndarray.astype(np.uint8)


def load_yaml(path):
    print(f'Load config from yaml file: {path}')
    with open(path, 'r') as f:
        return yaml.safe_load(f)


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


def setup_pretrained_models(device, species, condition):
    # preparing pix2pix model
    # config file for pix2pix setting
    pix2pix_model = []
    if not condition:
        cfg_path = './pix2pix/configs/pix2pix.yaml'
    else:
        cfg_path = './pix2pix/configs/pix2pix_phenotype.yaml'
    cfgs = load_yaml(cfg_path)
    pix2pix_leaf = create_model(cfgs, device)
    pix2pix_leaf.setup(cfgs, species, 'leaf')
    pix2pix_model.append(pix2pix_leaf)
    if not condition:
        pix2pix_branch = create_model(cfgs, device)
        pix2pix_branch.setup(cfgs, species, 'branch')
        pix2pix_model.append(pix2pix_branch)

    # preparing grounding_dino model
    ckpt_repo_id = 'ShilongLiu/GroundingDINO'
    ckpt_filename = 'groundingdino_swinb_cogcoor.pth'
    ckpt_config_filename = 'GroundingDINO_SwinB.cfg.py'
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename, device)

    # preparing mobile segment anything model
    sam_checkpoint = 'sam_vit_h_4b8939.pth'
    sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    return groundingdino_model, sam_predictor, pix2pix_model




