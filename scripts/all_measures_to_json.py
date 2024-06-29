import os
import json
import pandas as pd
from glob import glob

PATH = "masks"
OUTPUT_PATH = "data/all_measures.json"
folders = sorted(os.listdir(PATH), key=int)

measured_images = []
all_gt_measures = []

for f in folders:
    filepaths = glob(os.path.join(PATH, f, "*.png"))
    filenames = [os.path.basename(x) for x in filepaths]
    filenames.sort(key=lambda x: int(x[:-4]))
    
    measures = pd.read_csv(os.path.join(PATH, f, "metrics.csv"))
    measures = measures["length (px)"].tolist()
    
    if measures:
        measured_images.append(int(f))
        for filename, measure in zip(filenames, measures):
            all_gt_measures.append({"id": int(filename[:-4]), "measure": measure, "image_id": int(f)})

gt_measurement_data = {
    "measured_images": measured_images,
    "measures": all_gt_measures,
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(gt_measurement_data, f)