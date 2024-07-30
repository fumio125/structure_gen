#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/fundamentalvision/Deformable-DETR

CUDA_HOME="/usr/local/cuda-11.6" PATH="/usr/local/cuda-11.6/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-11.6/lib64/:$LD_LIBRARY_PATH"\
 sudo python3 setup.py build install
