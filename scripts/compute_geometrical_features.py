import os
import json
import cv2 as cv
import numpy as np
import pandas as pd

from lib.utils import poly_to_mask

JSON_PATH = "data/anns_w_measure.json"
DF_PATH = "data/eval_measuring.csv"
OUTPUT_PATH = "data/geometric_features.csv"

with open(JSON_PATH, "r") as f:
    all_anns = json.load(f)
imgs_data = all_anns["images"]
all_anns =  all_anns["annotations"]

# preprocess annotations to facilitate indexing
for key, anns in all_anns.items():
    ann_ids = [ann.pop("id") for ann in anns]
    anns = dict(zip(ann_ids, anns))
    all_anns[key] = anns
    
df = pd.read_csv(DF_PATH)

features = ["area", "perimeter", "circularity",
            "min_rect_length","min_rect_width",
            "aspect_ratio", "min_rect_area", "rectangularity",
            "convex-hull_area", "solidity"]
for ft in features:
    df[ft] = np.nan
    
for i,row in df.iterrows():
    img_id = row["image_id"]
    img_data = imgs_data[img_id]
    
    obj_id = row["id"]
    poly = all_anns[str(img_id)][obj_id]["segmentation"]
    mask = poly_to_mask(poly, (img_data["height"], img_data["width"]), True)
    mask = mask.astype(np.uint8)
    
    contours,_ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contour = max(contours, key=lambda x: x.shape[0])
    if contour.shape[0] <= 1: continue
    
    hull = cv.convexHull(contour)
    _,min_rect_dimensions,_ = cv.minAreaRect(contour)
    min_rect_dimensions = sorted(min_rect_dimensions)
 
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    rect_w, rect_l = min_rect_dimensions
    asp_ratio = rect_w / rect_l
    circularity = (4*np.pi*area) / (perimeter**2)
    rect_area = rect_w * rect_l
    rectangularity = area / rect_area
    hull_area = cv.contourArea(hull)
    solidity = area / hull_area
    
    values = [area, perimeter, circularity,
              rect_l, rect_w, asp_ratio, rect_area,
              rectangularity, hull_area, solidity]
    for col, value in zip(features, values):
        df.at[i, col] = value

df.to_csv(OUTPUT_PATH, index=False)