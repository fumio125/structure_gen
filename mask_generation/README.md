# Mask generation

## Enviromnment
* Blender 3.1 on Windows PC
* about whole proc
    * Python 3.8
    * about proc/texturing
        * docker: nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
        * torch: 1.13.1+cu116
        * torchvision: 0.14.1+cu116

## Usage
### 1. Download [Blender 3.1](https://download.blender.org/release/Blender3.1/)  
Note  
I don't know if it is compatible with other versions. Please refer to the official API

### 2. Install lsystem addon to Blender (details are [here](https://github.com/krljg/lsystem))
Note  
If you want to treat `utils.py` as a module, it must be started from `Blender Foundation` directory.  
* `addon` directroy is added to Blender `addon` directory  
* `modules` directory is added to Blender `modules` directory


### 3. Load `make_tamplate_segdata.py` into Blender and run it.  
Changes when you make data
```
# Below line 338 (if __name__ == '__main__'):

# line 341
DATA_TYPE: 'amodal' (including data for amodal segmentation) 
           'plant' (when you want to make plant-like plant)  
           'ara' (when you want to make Arabidopsis-like plant)
           'komatsuna' (when you want to make Komatsuna-like plant)

# line 346
root_dir: 'dir you want to save the data'

# line 354, 355, 362, 363
RENDER_NUM: Number of rendering per 3D plant model
RESOLUTIION: Resolution of rendered image 
RADIUS: Camera reloves around a circle with this radius (The recommended radius for each type is set now)
DATA_NUM: Number of times 3D plant models are generated
```  
More changes: Please correct the rules below line 62 of `make_template_segdata.py`
* The replacement rule of L-system (info is [here](http://algorithmicbotany.org/papers/abop/abop.pdf))
* How the leaves are attached to branch
    * Alternate (互生)  
      one leaf is attached to one node
    * Opposite (対生)  
      two leaves are attached to one node
    * Decussate opposite (十字対生)  
      two leaves are attached to one node,  
      but the direction in which the leaves are attached to the upper and lower nodes are 90 degrees different
    * Verticillate (輪生)
      four leaves (in this project) are attached to one node
        
### 4. Processing before texturing
Run as below
```
# 1. Conversion to frequently seen mask images
python bit_reverse.py

# 2. Make sure the organ is centered
python crop_mask.py
```

### 5. Texturing rendered images (species: yeddo hawthorn)
Here, I put a pre-trained yeddo hawthorn (車輪梅) organ generator ([pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)) , then run as below example
```
python texturing.py --gpu_id 0 --species hawthorn --render_num 10 --img_res 1024
```
Other options:
* phenotype: if you use `DATA_TYPE=='leaf'`, please add `--phenotype` option
* amodal: if you use `DATA_TYPE=='amodal'`, please add `--amodal` option








