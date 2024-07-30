# Leaf segmentation
***
This is a README for leaf segmentation  
## Grounded-SAM
***
preparation  
install [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [SegmentAnything](https://github.com/facebookresearch/segment-anything), [Detectron2](https://github.com/facebookresearch/detectron2) 
```
sudo pip3 install git+https://github.com/IDEA-Research/GroundingDINO.git
sudo pip3 install git+https://github.com/facebookresearch/segment-anything
sudo pip3 install git+https://github.com/facebookresearch/detectron2
```
**NOTE** when using command ``sudo``, please refer to [here](https://qiita.com/tks_00/items/bc04bc477d9019341859)

Then, run this script
```
python3 test.py --gpu_id <id> --species <species>
```
- gpu_id: Free gpu id (recommend)
- species: In case of Plant Phenomics, 'ara' or 'komatsuna' or 'hawthorn'
You can get qualitative and quantitative (AP scores) results

## MaskDINO
***
preparation  
install [Detectron2](https://github.com/facebookresearch/detectron2), ``requirements.txt`` and run ``make.sh``
```
sudo pip3 install git+https://github.com/facebookresearch/detectron2
sudo pip3 install -r requirements.txt
cd ./maskdino/modeling/pixel_decoder/ops/
sh make.sh
```
**NOTE** ``make.sh`` is run with ``python3``, so if you use ``python``, please rewrite ``make.sh``

Then, run this script when you train the model
```
CUDA_VISIBLE_DEVICES='free gpu ids' python3 train_net.py --num-gpus <number of gpus you use> \
--config-file <path to the config file> \
--species <species> --version <expriment version> \
opts (ex. SOLVER.IMS_PER_BATCH 16)
```
When you test the model, you run this script
```
python3 test.py --config_file <path to the config file> --species <species name> --gpu_id <one gpu id> \ 
--threshold <threshold of area> MODEL.WEIGHTS /path/to/ckpt_file 
```
**NOTE**
My ckpt files are in ``ckpt`` dirs
However, there is only an ``ara`` file in this dir (I retrain Komatsuna and look for Hawthorn best ckpt)
