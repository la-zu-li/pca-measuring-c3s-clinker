import os
import json

PATH = "images test"
OUTPUT_PATH = "data"
with open(os.path.join(PATH, "_annotations.coco.json"), "r") as f:
    anns_dict = json.load(f)
img_data = anns_dict["images"]
anns = anns_dict["annotations"]

with open("all_measures.json", "r") as f:
    all_measures_dict = json.load(f)
measured_images = all_measures_dict["measured_images"]
measures = all_measures_dict["measures"]

# merge annotations with measures
anns_measured = [ann for ann in anns if ann["image_id"] in measured_images]
anns_w_measure = dict([(im_id, []) for im_id in measured_images])

for ann, measure in zip(anns_measured, measures):
    if ann["id"] != measure["id"]:
        raise ValueError("Annotations and measures should have same ID")
    
    img_id = ann["image_id"]
    ann_w_measure = {"id": ann["id"], "segmentation": ann["segmentation"], "measure": measure["measure"]}
    anns_w_measure[img_id].append(ann_w_measure)

anns_w_measure_n_images = {"images": img_data, "annotations": anns_w_measure}
with open(os.path.join(OUTPUT_PATH, "anns_w_measure.json"), "w") as f:
    json.dump(anns_w_measure_n_images, f)