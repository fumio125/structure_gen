# Dataset
My data is in 

## Data structure
- pix2pix: Data for pix2pix (organ generation) training
- segmentation: Data for leaf segmentation training
- src: original data (Blender's mask image and real plant image)
- unetGAN: Data for unetGAN training

## preparation
you copy data generated with Blender to ``src/syn_plant/<species name>``

## code  
### src 
``proc_data.py`` (in ``syn_plant`` dir)
**How to run**
```
python3 proc_data.py --species <species name> --render_times <render times per one model>
```
This allows the copied data to be used for various task 

### pix2pix  
``make_dataset.py`` (it uses Grounded-SAM)  
**How to run**
```
python3 make_dataset.py --img_res <image resolution> --text_prompt <text> \
--device_id <free gpu id> --mode <train or test> --species <species name>
```

### segmentation  
``visualization.py`` (in ``GT``dir, for result visualization, it uses detectron2)  
**How to run**
```
python3 visualization.py --species <species name>
```
``coco.py`` (in ``segmentation`` dir, for creating coco format)  
**How to run**
```
python3 coco.py --species <species name> (--test; if you make coco format for test data)
```



