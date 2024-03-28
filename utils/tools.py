import cv2
import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch import Tensor


def find_contours(mask: Tensor):
    h,w = mask.shape
    if isinstance(mask, Tensor): 
        mask = mask.numpy().astype(np.uint8)
    edge = np.zeros((h,w), dtype=np.uint8)

    contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    assert len(contours) == 1
    contours = contours[0].squeeze()
    edge[contours[:,1], contours[:,0]] = 1

    # check
    assert edge.sum() == (edge*mask).sum()
    return edge


def find_contour_points(mask: Tensor):
    '''
        mask: (h,w), 0 or 1
        return: contours (n,2)
                the x,y of the points 
    '''
    h,w = mask.shape
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy().astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    edge = np.zeros((h,w), dtype=np.uint8)

    contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    # assert len(contours) == 1
    if len(contours) != 1:
        return np.array([])
    
    contours = contours[0].squeeze()
    edge[contours[:,1], contours[:,0]] = 1

    # check
    assert edge.sum() == (edge*mask).sum()
    return contours


def hausdorff_distance(mask1: Tensor, mask2: Tensor, percentile: int = 95):
    if isinstance(mask1, torch.Tensor) and mask1.device == 'cuda':
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()

    contours1 = find_contour_points(mask1)
    contours2 = find_contour_points(mask2)
    if contours1.size == 0 or contours2.size == 0:
        return 0

    dist = cdist(contours1, contours2)
    dist = np.concatenate((np.min(dist, axis=0), np.min(dist, axis=1)))
    assert percentile >= 0 and percentile <= 100, 'percentile invaild'
    hausdorff_dist = np.percentile(dist, percentile)

    return hausdorff_dist
