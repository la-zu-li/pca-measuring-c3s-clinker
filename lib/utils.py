import numpy as np
import cv2 as cv

# scale of micrometers per pixel used on studied clinker images
SCALE_UM_BY_PX = 0.3105590062111801

def px_to_micrometers(distance_pixels, scale=SCALE_UM_BY_PX):
    distance_in_micrometers = distance_pixels * scale
    return distance_in_micrometers

def micrometers_to_px(distance_micrometers, scale=SCALE_UM_BY_PX):
    distance_in_pixels = distance_micrometers / scale
    return distance_in_pixels

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

def split_img_rec(img, iter=3):
    if iter<=0:
        return [img]

    img = np.rot90(img)
    half = img.shape[0] // 2
    b,a = img[:half], img[half:]

    return split_img_rec(a, iter-1) + split_img_rec(b, iter-1)

def apply_closing_3dmasks(masks, ksize=5, iter=1):

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ksize, ksize))
    closing = lambda x: cv.morphologyEx(x, cv.MORPH_CLOSE, kernel, iterations=iter)

    masks = masks.astype(np.uint8)
    masks = np.stack([closing(x) for x in masks])
    masks = masks.astype(bool)
    return masks

def calc_centroid(blob):
    m = cv.moments(blob)
    centroid_x = int(m["m10"] / m["m00"])
    centroid_y = int(m["m01"] / m["m00"])
    return (centroid_x, centroid_y)

def contour_of_blob(blob):
    contours,_ = cv.findContours(blob, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cont_size = lambda x: x.shape[0]
    contour = max(contours, key=cont_size)
    return contour

def as_bw_images(mask3d):
    mask3d_uint = mask3d.astype(np.uint8)
    mask3d_bw = mask3d_uint * 255
    bw_images = [img for img in mask3d_bw]
    return bw_images
