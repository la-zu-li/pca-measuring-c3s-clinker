import numpy as np
import cv2 as cv

def poly_to_mask(polygon, image_size, bool_mask=False) -> np.ndarray:
    mask = np.zeros(image_size, dtype=np.uint8)
    polygon_points = np.array(polygon).reshape(-1, 2).astype(np.int32)
    cv.fillPoly(mask, [polygon_points], 255)
    if bool_mask: mask = mask > 0
    return mask

def iou_masks(mask1: np.ndarray, mask2: np.ndarray) -> float:
    mask1_area = np.count_nonzero(mask1)
    mask2_area = np.count_nonzero(mask2)
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    iou = intersection/(mask1_area+mask2_area-intersection)
    return iou

def all_iou_combinations(masks1: list[np.ndarray], masks2: list[np.ndarray]) -> np.ndarray:   
    n_masks1 = len(masks1)
    n_masks2 = len(masks2)
    
    ious = np.zeros((n_masks1, n_masks2), dtype=np.float64)
    for i in range(n_masks1):
        for j in range(n_masks2):
            ious[i,j] = iou_masks(masks1[i], masks2[j])
    return ious