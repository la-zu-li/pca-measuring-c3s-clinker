import math
import numpy as np
import cv2 as cv

import itertools
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks, argrelmax

def longest_diagonal_pca(points):

    pca = PCA(n_components=1)
    pca.fit(points)
    principal_component = pca.components_[0]
    points_pca = np.dot(points, principal_component)

    start_index = np.argmin(points_pca)
    end_index = np.argmax(points_pca)
    start, end = points[start_index], points[end_index]
    start, end = start.tolist(), end.tolist()
    length = math.dist(start, end)

    return start, end, length

def longest_diagonal_lr(points):

    lr = LinearRegression()
    lr.fit(points[:, :1], points[:, 1])
    vector = np.array([1, lr.coef_[0]])
    points_dotted = np.dot(points, vector)

    start_index = np.argmin(points_dotted)
    end_index = np.argmax(points_dotted)
    start, end = points[start_index], points[end_index]
    start, end = start.tolist(), end.tolist()
    length = math.dist(start, end)

    return start, end, length

def longest_diagonal_centroid(points, starting_frac=3, step=0.5, use_find_peaks=True):

    m = cv.moments(points)
    # centroid = (m['m10'] / (m['m00'] + 1e-5), m['m01'] / (m['m00'] + 1e-5))
    centroid = (m['m10']/m['m00'], m['m01']/m['m00'])

    distances = np.array([math.dist(centroid, x) for x in points])

    # find local maxima using tolerance criterium, based on lateral distance
    n_points = distances.shape[0]
    min_distance_frac = starting_frac
    if find_peaks:
        local_max_indexes,_ = find_peaks(distances, distance=n_points//min_distance_frac)
    else:
        local_max_indexes = argrelmax(distances, mode='wrap', order=int(n_points//min_distance_frac))[0]

    # decrease distance tolerance for local maxima until at least two are found
    while len(local_max_indexes) < 2:
        min_distance_frac += step
        if find_peaks:
            local_max_indexes,_ = find_peaks(distances, distance=n_points//min_distance_frac)
        else:
            local_max_indexes = argrelmax(distances, mode='wrap', order=int(n_points//min_distance_frac))[0]

    local_max_vals = distances[local_max_indexes]

    i_2biggest_local_max_indexes = np.argsort(local_max_vals)[-2:]
    i_2nd_biggest_peak, i_biggest_peak = local_max_indexes[i_2biggest_local_max_indexes]

    start, end = points[i_biggest_peak], points[i_2nd_biggest_peak]
    start, end = start.tolist(), end.tolist()
    length = math.dist(start, end)

    return start, end, length

def farthest_pair_of_points_brute_force(points):
    point_pairs = itertools.combinations(points, 2)
    
    max_distance = 0
    farthest_points = None
    for pair in point_pairs:
        distance = math.dist(*pair)
        if distance > max_distance:
            max_distance = distance
            farthest_points = pair

    points = [x.tolist() for x in farthest_points]
    return *points, max_distance