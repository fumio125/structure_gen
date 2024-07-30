# Plant image generation

This is a README for plant image generation

## Pix2pix
preparation  
install [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [SegmentAnything](https://github.com/facebookresearch/segment-anything) (use for mask and image pair dataset)  
Then, install ``requirements.txt``
```
sudo pip3 install git+https://github.com/IDEA-Research/GroundingDINO.git
sudo pip3 install git+https://github.com/facebookresearch/segment-anything
sudo pip3 install -r requirements.txt
```
**NOTE** when using command ``sudo``, please refer to [here](https://qiita.com/tks_00/items/bc04bc477d9019341859)

Then, run this script
```
python3 train.py --dataroot <path to the data dir> --name <experiment_name> --gpu_ids <free gpu id>\
--model pix2pix --netG unet_256 (unet_128, in case of Arabidopsis, Komatsuna) \
--direction AtoB --lambda_l1 100 --norm batch --display_id 0 --display_freq 400 \
--preprocess none --no_flip (--use_wandb if you use weight and bias) 
```
**NOTE** 
- Your training log is stored ``checkpoints/args.name``
- When you test pix2pix, please check below explanation.
- Please copy checkpoint to ``unetGAN/pix2pix/ckpt`` (as a default, my ckpt is stored there, so please rename that)

## UnetGAN
preparation  
install [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [SegmentAnything](https://github.com/facebookresearch/segment-anything) (use for mask and image pair dataset)  
Then, install ``requirements.txt`` in Pix2pix
```
sudo pip3 install git+https://github.com/IDEA-Research/GroundingDINO.git
sudo pip3 install git+https://github.com/facebookresearch/segment-anything
sudo pip3 install -r requirements.txt
```
0. When you want to test pix2pix ability, you run this script
```
python3 test_pix2pix.py --gpu_id <free gpu id> --src_path <path to the data dir> --tgt_path <path to the result dir> \
--text <leaf or branch> --species <species name> --phenotype <when testing 'ara' or 'komatsuna', you add this argument>
```
1. Make dataset for unetGAN
```
# for training
python3 proc_realworld.py --gpu_id <free gpu id> --res <image resolution> --species <species name> \
--phenotype <when testing 'ara' or 'komatsuna', you add this argument> --type proc
# for test
python3 proc_synthesize.py --gpu_id <free gpu id> --species <species name> --render_num <render times per one model> \
--phenotype <when testing 'ara' or 'komatsuna', you add this argument> --img_res <image resolution>
```
- type: For journal's figure, you specify 'thesis'

2. Train UnetGAN
```
python3 train.py --train_epoch 500 --date <yymmdd_vxx> --gpu_id <free gpu id> (--resume <epoch>) \
--species <species name> --train_res <image resolution> --weight_adv 0.01
```
**NOTE**
- Your training log is stored ``checkpoints/args.species+args.date``

3. test UnetGAN
```
python3 test_unetgan.py --load_e <epoch you want to test (int)> --gpu_id <free gpu id> --species <species name> \
--test_res <image resolution> --train_version <yymmdd_vxx> --phenotype <when testing 'ara' or 'komatsuna', you add this argument> --img_res <image resolution>
```
**NOTE**
- The result is stored ``/data/segmentation/p_args.species/train/img`` in docker
