# Script for generating polygons from binary masks

import os
import json
from glob import glob

from utils import poly_to_mask

IMAGES_PATH = "images test"
OUTPUT_PATH = "masks"

imgs_paths = glob(os.path.join(IMAGES_PATH, "*jpg"))
ann_path = os.path.join(IMAGES_PATH, "_annotations.coco.json")

with open(ann_path, "r") as f:
    anns = json.load(f)

for im_dict in anns["images"]:
    img_id = im_dict["id"]
    folder_path = os.path.join(OUTPUT_PATH, str(img_id))

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

for ann in anns["annotations"]:
    im_id = ann["image_id"]
    im_dict = anns["images"][im_id]

    id = ann["id"]
    poly = ann["segmentation"]
    
    im_shape = (im_dict["height"], im_dict["width"])
    mask = poly_to_mask(poly, im_shape)

    cv.imwrite(os.path.join(OUTPUT_PATH, str(im_id), f"{id}.png"), mask)