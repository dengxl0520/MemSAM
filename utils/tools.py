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


def corr(x, y):
    '''
        x : gt
        y : pred
        A = mean( (y_real - mean(y_real)) * (y_predict - mean(y_predict)) )
        B = std(y_real) * std(y_predict)
        corr = A / B
    '''
    A = ((x - x.mean()) * (y - y.mean())).mean()
    B = std(x) * std(y)
    corr = A / B
    return corr

def bias(x, y):
    '''
        x : gt
        y : pred
        bias = sum( abs(y_real - y_predict) ) / len( y_real )
    '''
    return (abs(x - y)).mean()

def std(x):
    '''
        A = (y - mean(y)) * (y - mean(y))
        std = sqrt( sum(A) / n )
    '''
    # A = ((x - x.mean()) * (x - x.mean())).mean()
    # std = np.sqrt(A)
    return np.std(x)

def draw_sem_seg_by_cv2_sum(image, gt_sem_seg, pred_sem_seg, palette):
    '''
        image: [3,h,w] numpy.ndarray
        gt_sem_seg: [h,w] numpy.ndarray
        pred_sem_seg: [h,w] numpy.ndarray
        palette: [bg, gt, pred, overlap] numpy.ndarray
    '''
    gt_sem_seg = gt_sem_seg.astype(np.uint8)
    pred_sem_seg = pred_sem_seg.astype(np.uint8)
    mask = 2 * pred_sem_seg + gt_sem_seg
    mask = mask.squeeze()
    
    ids = np.unique(mask)
    color_mask = np.zeros_like(image)
    for idx in ids:
        color_mask[0][mask == idx] = palette[idx][0]
        color_mask[1][mask == idx] = palette[idx][1]
        color_mask[2][mask == idx] = palette[idx][2]
    
    results = cv2.addWeighted(image, 0.2, color_mask, 0.8, 0)
    # for i in range(mask.shape[0]):
    #     for j in range(mask.shape[1]):
    #         if mask[i,j] !=0:
    #             image[...,i,j] = results[...,i,j]
    mask = np.expand_dims(mask, 0).repeat(3,axis=0) # np
    image[mask != 0] = results[mask != 0]
    return image
