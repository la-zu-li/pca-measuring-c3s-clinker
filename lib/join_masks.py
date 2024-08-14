import numpy as np
import cv2 as cv

# computes IoU for 1D arrays
def iou_arrays(arr1, arr2):
    intersection_set = np.intersect1d(arr1, arr2, assume_unique=True)
    intersection = intersection_set.shape[0]
    union = arr1.shape[0] + arr2.shape[0] - intersection

    return intersection / union

# computes IoU for lists of 1D Arrays
def all_iou_arrays(arrays1, arrays2):
    n_arr1, n_arr2 = len(arrays1), len(arrays2)

    ious = np.zeros((n_arr1, n_arr2), dtype=np.float64)
    for i in range(n_arr1):
        for j in range(n_arr2):
            arr1, arr2 = arrays1[i], arrays2[j]
            ious[i,j] = iou_arrays(arr1, arr2)

    return ious

def expand_masks_size(l_mask, r_mask):
    r_fill = np.zeros(l_mask.shape[:1] + r_mask.shape[1:], dtype=bool)
    l_fill = np.zeros(r_mask.shape[:1] + l_mask.shape[1:], dtype=bool)

    l_mask_concat = np.concatenate((l_mask, r_fill), axis=2)
    r_mask_concat = np.concatenate((l_fill, r_mask), axis=2)

    return l_mask_concat, r_mask_concat

def join_crops_masks(l_mask, r_mask, iou_thresh):
    # print(l_mask.shape, r_mask.shape)
    # select masks that are on the limit between the crops
    idxs, borders = np.where(l_mask[:,:,-1])
    border_masks_idxs_l, splits = np.unique(idxs, return_index=True)
    l_borders = np.split(borders, splits[1:])

    idxs, borders = np.where(r_mask[:,:, 0])
    border_masks_idxs_r, splits = np.unique(idxs, return_index=True)
    r_borders = np.split(borders, splits[1:])

    # computes IoU between two borders
    ious = all_iou_arrays(l_borders, r_borders)

    max_iou_idx = ious.argmax(axis=1)
    max_iou = ious[range(max_iou_idx.shape[0]), max_iou_idx]

    # remove matches that are not good enough
    true_matches = max_iou > iou_thresh
    max_iou_idx = max_iou_idx[true_matches]

    border_masks_idxs_l = border_masks_idxs_l[true_matches]
    corresp_border_masks_idxs_r = border_masks_idxs_r[max_iou_idx]

    l_mask, r_mask = expand_masks_size(l_mask, r_mask)
    l_masks_list = [m for m in l_mask]
    r_masks_list = [m for m in r_mask]

    # merge matched masks
    for i,j in zip(border_masks_idxs_l, corresp_border_masks_idxs_r):
        left_mask = l_masks_list[i]
        right_mask = r_masks_list[j]
        l_masks_list[i] = np.logical_or(left_mask, right_mask)

    # remove redundant masks
    for j in sorted(corresp_border_masks_idxs_r, reverse=True):
        r_masks_list.pop(j)

    # stack everything as a single mask
    all_masks_list = l_masks_list + r_masks_list
    all_masks = np.stack(all_masks_list)

    return all_masks

# join the masks recursively. Works for crops of images split recursively.
def join_crops_masks_rec(crops_masks: list[np.ndarray], iou_thresh=0.5):
    n_crops = len(crops_masks)

    if n_crops <= 1:
        return crops_masks[0]

    l = join_crops_masks_rec(crops_masks[:n_crops//2])
    r = join_crops_masks_rec(crops_masks[n_crops//2:])

    l = np.rot90(l, axes=(2,1))
    r = np.rot90(r, axes=(2,1))

    joined_mask = join_crops_masks(l,r, iou_thresh)
    return joined_mask