import cv2 as cv
import numpy as np
import pandas as pd
import pickle as pkl
from time import time

from lib.measure_methods import (
    longest_diagonal_pca as pca_method,
    longest_diagonal_centroid as centroid_method,
    longest_diagonal_lr as linear_regression,
    farthest_pair_of_points_brute_force as brute_force
)
PKL_PATH = "dt_masks.pkl"
DF_PATH = "data/eval_data.csv"
OUTPUT_PATH = "data/eval_measuring.csv"

with open(PKL_PATH, "rb") as f:
    all_dt_masks = pkl.load(f)
    
df = pd.read_csv(DF_PATH)

name_methods = ["pca", "centroid", "lr", "brute-force"]
methods = [pca_method, centroid_method, linear_regression, brute_force]
kwargs_methods = [{}, {}, {}, {}]

# creating and initializing new columns
df["find-contour_time"] = np.nan
for method in name_methods:
    df[f"{method}_measure"] = np.nan
    df[f"{method}_time"] = np.nan
    
for i,row in df.iterrows():
    img_id = row["image_id"]
    dt_mask_id = row["detected_mask_id"]
    
    if dt_mask_id < 0: continue
    
    dt_mask = all_dt_masks[img_id][dt_mask_id]
    
    start = time()
    contours,_ = cv.findContours(dt_mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contour = max(contours, key=lambda x: x.shape[0])
    points = contour.squeeze()
    end = time()
    df.at[i,"find-contour_time"] = end-start
    
    if len(points.shape) != 2: continue
    
    for name, method, kwargs in zip(name_methods, methods, kwargs_methods):        
        start = time()
        _,_,length = method(points, **kwargs)
        end = time()
        df.at[i,f"{name}_time"] = end-start
        df.at[i,f"{name}_measure"] = length

df.to_csv(OUTPUT_PATH, index=False)