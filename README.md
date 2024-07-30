# Structure-aware Image Generation

- [Abstract Usage](##Abstract Usage)
  - [Docker Usage](###Docker Usage)

### Introduction
This project page is a multi-project portal 

- mask generation: automatic CG plant mask generation with Blender
- plant generation: automatic synthetic plant image generation with Pix2pix and U-Net
- leaf segmentation: segmentation of leaf with ours or Grounded-SAM

Each project has README.md, so please refer to each page for details such as how to run the code
## Abstract Usage
These projects are run in the following environment :

- mask generation: Windows PC which has Intel i9 13900K CPU, RTX 4900 GPU and 64GB of main memory and Blender 3.1
- plant generation and leaf segmentation: Docker container (FROM [nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04](https://hub.docker.com/layers/nvidia/cuda/11.6.2-cudnn8-devel-ubuntu20.04/images/sha256-4eeb683bf695d431ecba6c949b4ee86c1cff61c2786c4de93b8df095f0852b78?context=explore)), python 3.8.10, Pytorch 1.13.1+cu116 and torchvision 0.14.1+cu116

### Docker Usage
1. Build image from Dockerfile
```
sh build.sh <title> 
```
- title: image name (Feel free to decide)
2. Run image
```
sh run.sh <title> <container_id>
```
- title: image name you specified when building image
- container_id (basically, 0 or 1)

**NOTE**
The running container only includes PyTorch and torchvision, so please refer to the README of each project to install the other requirements.


### Dependency
This project depends on [Segment Anything](https://github.com/facebookresearch/segment-anything), [MaskDINO](https://github.com/IDEA-Research/MaskDINO), [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), and [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
