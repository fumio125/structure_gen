import cv2
import torch
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as F

import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import load_image, predict


# rewriting later
def process_grounding_dino(model, img_path, res, text, device):
    # print('processing with grounding dino...')
    if text == 'branch':
        box_threshold = 0.2
        text_threshold = 0.2
    else:
        box_threshold = 0.28
        text_threshold = 0.25

    image_source, image = load_image(img_path)
    if image_source.shape[0] != res:
        transform = transforms.Resize((res, res))
        image_source, image = cv2.resize(image_source, (res, res)), transform(image)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    return image_source, torch.tensor(image_source).to(device), boxes


def process_segment_anything(model, image_source, bb, device):
    # print('processing with segment anything...')

    model.set_image(image_source)
    H, W, _ = image_source.shape

    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(bb) * torch.Tensor([W, H, W, H])
    transformed_boxes = model.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
    if transformed_boxes.shape[0] == 0:
        return None

    masks, _, _ = model.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    return masks

def create_organ_replace_img(model1, model2, model3, img_path, res, text, device):
    '''
    Description of each size
    ・image_size: the size of original image (=image_source.shape[0])
    ・mask_size: the size which we want to use for processing (about image_source / 4)
    ・pix2pix_size: (256, 256) (train pix2pix with this size)
    ・crop_size: the size of cropped image
    '''
    image_source, processed_img, boxes = process_grounding_dino(model=model1, img_path=img_path, res=res, text=text, device=device)
    # power_po_img = processed_img
    masks = process_segment_anything(model=model2, image_source=image_source, bb=boxes, device=device)
    if masks is None:
        return np.array(image_source), processed_img.cpu().numpy()
    image_size = image_source.shape[0]
    pix2pix_size = (128, 128)
    for leaf_id in range(len(masks)):

        box = boxes[leaf_id].cpu().numpy() * image_size

        square = int(max(box[2], box[3]) / 2) + 1
        x1, x2 = max(int(box[0]) - square, 0), min(int(box[0]) + square, image_size)
        y1, y2 = max(int(box[1]) - square, 0), min(int(box[1]) + square, image_size)

        crop_w, crop_h = x2 - x1, y2 - y1

        if text == 'leaf':
            if crop_w == crop_h:
                crop_size = crop_w
                # make pix2pix input image
                leaf_mask_cropped = masks[leaf_id][0][y1:y2, x1:x2]
                leaf_mask_real = F.resize(img=leaf_mask_cropped.unsqueeze(0), size=pix2pix_size,
                                          interpolation=transforms.InterpolationMode.NEAREST, antialias=False)
                leaf_mask_real = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(
                    leaf_mask_real.to(torch.float32).repeat(3, 1, 1))
                # generate paste image (fake leaf image)
                model3.set_input(leaf_mask_real.unsqueeze(0))
                model3.test()
                img_dict = model3.get_current_visuals()
                leaf_img_fake = img_dict['fake_B'][0]
                # process fake leaf image
                leaf_img_fake = (leaf_img_fake + 1) / 2.0 * 255.0
                leaf_img_fake = F.resize(leaf_img_fake, size=(crop_size, crop_size))
                leaf_img_fake = torch.masked_fill(input=leaf_img_fake, mask=~leaf_mask_cropped, value=0)
                # paste fake image on original image
                img1 = torch.masked_fill(input=torch.permute(processed_img[y1:y2, x1:x2], (2, 0, 1)), mask=leaf_mask_cropped, value=0)
                img2 = torch.bitwise_or(img1, leaf_img_fake.to(torch.uint8))

                processed_img[y1:y2, x1:x2] = torch.permute(img2, (1, 2, 0))

        elif text == 'branch':
            if crop_w == crop_h and crop_w < mask_size:
                crop_size = crop_w
                # make pix2pix input image
                branch_mask_cropped = masks[leaf_id][0][y1:y2, x1:x2]
                branch_mask_real = F.resize(img=branch_mask_cropped.unsqueeze(0), size=pix2pix_size,
                                          interpolation=transforms.InterpolationMode.NEAREST, antialias=False)
                branch_mask_real = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(
                    branch_mask_real.to(torch.float32).repeat(3, 1, 1))
                # generate paste image (fake branch image)
                model4.set_input(branch_mask_real.unsqueeze(0))
                model4.test()
                img_dict = model4.get_current_visuals()
                branch_img_fake = img_dict['fake_B'][0]
                # process fake leaf image
                branch_img_fake = (branch_img_fake + 1) / 2.0 * 255.0
                branch_img_fake = F.resize(branch_img_fake, size=(crop_size, crop_size))
                branch_img_fake = torch.masked_fill(input=branch_img_fake, mask=~branch_mask_cropped, value=0)
                # paste fake image on original image
                img1 = torch.masked_fill(input=torch.permute(processed_img[y1:y2, x1:x2], (2, 0, 1)),
                                         mask=branch_mask_cropped, value=0)
                img2 = torch.bitwise_or(img1, branch_img_fake.to(torch.uint8))
                processed_img[y1:y2, x1:x2] = torch.permute(img2, (1, 2, 0))

    return np.array(image_source), processed_img.cpu().numpy()


def create_thesis_data(model1, model2, model3, img_path, res, text, device):
    '''
    Description of each size
    ・image_size: the size of original image (=image_source.shape[0])
    ・mask_size: the size which we want to use for processing (about image_source / 4)
    ・pix2pix_size: (256, 256) (train pix2pix with this size)
    ・crop_size: the size of cropped image
    '''
    image_source, processed_img, boxes = process_grounding_dino(model=model1, img_path=img_path, res=res, text=text, device=device)
    power_po_img = processed_img
    masks = process_segment_anything(model=model2, image_source=image_source, bb=boxes, device=device)
    if masks is None:
        return np.array(image_source), processed_img.cpu().numpy()
    image_size = image_source.shape[0]
    # mask_size = res / 4
    pix2pix_size = (128, 128)
    for leaf_id in range(len(masks)):

        box = boxes[leaf_id].cpu().numpy() * image_size

        square = int(max(box[2], box[3]) / 2) + 1
        x1, x2 = max(int(box[0]) - square, 0), min(int(box[0]) + square, image_size)
        y1, y2 = max(int(box[1]) - square, 0), min(int(box[1]) + square, image_size)

        crop_w, crop_h = x2 - x1, y2 - y1

        power_po_res_img = torch.tensor(np.array(image_source))
        if text == 'leaf':
            if crop_w == crop_h:
                crop_size = crop_w
                # make pix2pix input image
                leaf_mask_cropped = masks[leaf_id][0][y1:y2, x1:x2]
                cv2.imwrite(f'./tmp/real/i_mask_{leaf_id}.png', cv2.cvtColor(leaf_mask_cropped.cpu().numpy().astype(np.uint8)*255, cv2.COLOR_BGR2RGB))
                leaf_mask_real = F.resize(img=leaf_mask_cropped.unsqueeze(0), size=pix2pix_size,
                                          interpolation=transforms.InterpolationMode.NEAREST, antialias=False)
                leaf_mask_real = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(
                    leaf_mask_real.to(torch.float32).repeat(3, 1, 1))
                # generate paste image (fake leaf image)
                model3.set_input(leaf_mask_real.unsqueeze(0))
                model3.test()
                img_dict = model3.get_current_visuals()
                leaf_img_fake = img_dict['fake_B'][0]
                # process fake leaf image
                leaf_img_fake = (leaf_img_fake + 1) / 2.0 * 255.0
                cv2.imwrite(f'./tmp/real/i_crop.png_{leaf_id}.png', cv2.cvtColor(torch.permute(leaf_img_fake, (1, 2, 0)).cpu().numpy(), cv2.COLOR_BGR2RGB))
                leaf_img_fake = F.resize(leaf_img_fake, size=(crop_size, crop_size))
                leaf_img_fake = torch.masked_fill(input=leaf_img_fake, mask=~leaf_mask_cropped, value=0)
                img0 = torch.masked_fill(input=torch.permute(power_po_res_img[y1:y2, x1:x2].to(device), (2, 0, 1)), mask=~leaf_mask_cropped, value=0)
                cv2.imwrite(f'./tmp/real/i_real_{leaf_id}.png', cv2.cvtColor(torch.permute(img0, (1, 2, 0)).cpu().numpy(), cv2.COLOR_BGR2RGB))
                cv2.imwrite(f'./tmp/real/i_fake_{leaf_id}.png', cv2.cvtColor(torch.permute(leaf_img_fake, (1, 2, 0)).cpu().numpy(), cv2.COLOR_BGR2RGB))
                # paste fake image on original image
                img1 = torch.masked_fill(input=torch.permute(processed_img[y1:y2, x1:x2], (2, 0, 1)), mask=leaf_mask_cropped, value=0)
                img3 = torch.masked_fill(input=torch.permute(power_po_res_img[y1:y2, x1:x2].to(device), (2, 0, 1)), mask=leaf_mask_cropped, value=0)
                print(img1.shape)
                cv2.imwrite(f'./tmp/real/i_base_{leaf_id}.png', cv2.cvtColor(torch.permute(img1, (1, 2, 0)).cpu().numpy(), cv2.COLOR_BGR2RGB))
                img2 = torch.bitwise_or(img1, leaf_img_fake.to(torch.uint8))
                img4 = torch.bitwise_or(img3, leaf_img_fake.to(torch.uint8))
                cv2.imwrite(f'./tmp/real/i_paste_{leaf_id}.png', cv2.cvtColor(torch.permute(img2, (1, 2, 0)).cpu().numpy(), cv2.COLOR_BGR2RGB))

                print(power_po_res_img.shape)
                power_po_res_img[y1:y2, x1:x2] = torch.permute(img3, (1, 2, 0))
                cv2.imwrite(f'./tmp/real/input_{leaf_id}.png', cv2.cvtColor(power_po_res_img.numpy(), cv2.COLOR_BGR2RGB))

                power_po_res_img[y1:y2, x1:x2] = torch.permute(img4, (1, 2, 0))
                cv2.imwrite(f'./tmp/real/real_{leaf_id}.png', cv2.cvtColor(power_po_res_img.numpy(), cv2.COLOR_BGR2RGB))






