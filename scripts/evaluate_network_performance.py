import os
import json
import numpy as np
import pandas as pd
import pickle as pkl
from utils import all_iou_combinations

OUTPUT_PATH = "data"
JSON_PATH = "data"
PKL_PATH = ""
IOU_THRESH = 0.5 # threshold used to validate if it's a hit or not

with open(os.path.join(JSON_PATH, "anns_w_measure.json"), "r") as f:
    anns = json.load(f)

with open(os.path.join(PKL_PATH, "dt_masks.pkl"), "rb") as f:
    all_dt_masks = pkl.load(f)

gt_anns_measures = anns["annotations"]
filename_and_id = [[x["id"], x["file_name"]] for x in anns["images"]]
id_to_filename = dict(filename_and_id)

all_eval_data = []
total_fps = 0
total_fns = 0
for img_id, anns_measures in list(gt_anns_measures.items()):
    img_id = int(img_id)
    img_data = imgs_data[img_id]
    height, width = img_data["height"], img_data["width"]
    
    ids = [x["id"] for x in anns_measures]
    gt_poly = [x["segmentation"] for x in anns_measures]
    gt_measures = [x["measure"] for x in anns_measures]
    gt_masks = [poly_to_mask(x, (height, width), True) for x in gt_poly]
    dt_masks = all_dt_masks[img_id]

    ious_mx = all_iou_combinations(gt_masks, dt_masks)
    ious_mx_hit = ious_mx >= IOU_THRESH
    
    hit_sum_gt = ious_mx_hit.sum(axis=1)
    hit_sum_dt = ious_mx_hit.sum(axis=0)
    
    fns = np.where(hit_sum_gt == 0)
    tps = np.where(hit_sum_gt > 0)
    
    # increase total number of false positives and negatives
    total_fps += np.count_nonzero(hit_sum_dt == 0)
    total_fps += np.count_nonzero(hit_sum_gt > 1)
    total_fns += len(fns)

    gt_ious = ious_mx.max(axis=1)
    gt_ious[fns] = -1
    corresp_dt_index = ious_mx.argmax(axis=1)
    corresp_dt_index[fns] = -1
    
    gt_areas = [x.sum() for x in gt_masks]
    dt_areas = [dt_masks[i].sum() if i>=0 else -1 for i in corresp_dt_index]
    
    # append data
    filename = id_to_filename[img_id]
    img_ids = [img_id for _ in anns_measures]
    filenames = [filename for _ in anns_measures]
    data = list(zip(filenames, img_ids, ids, gt_areas, gt_measures, corresp_dt_index, dt_areas, gt_ious))
    all_eval_data.extend(data)
    
df = pd.DataFrame(all_eval_data, columns=[
    "filename", "image_id", "id", "area", "ground-truth_measure", 
    "detected_mask_id", "detected_mask_area", "iou"])
df_summary = pd.DataFrame({"total_FP": total_fps, "total_FN":total_fns})

df.to_csv(os.path.join(OUTPUT_PATH, "eval_data.csv"))
df_summary.to_csv(os.path.join(PATH, "data_summary.json"))