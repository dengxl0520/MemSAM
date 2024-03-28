import os
from random import randint
import numpy as np
import torch
from skimage import io, color
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
import torch.nn.functional as nnF
from typing import Callable
import os
import cv2
import pandas as pd
from numbers import Number
from typing import Container
from collections import defaultdict
from batchgenerators.utilities.file_and_folder_operations import *
from collections import OrderedDict
from torchvision.transforms import InterpolationMode
from einops import rearrange
import random


def load_video_and_mask_file(img_path: str,
                             anno_path: str,
                             frame_length: int = 10):
    '''
        modify by ultrasound_npy_sequence_load() from EchoGraphs
        args:
            img_path: videos .npy file path
            anno_path: annotations .npz file path
            frame_length: Placing ED frame at index 0, and ES frame at index -1
        return: 
            imgs: (F,3,112,112)
            masks: (2,112,112)  ED and ES frame
            ef: float
            edv, esv, spacing
    '''
    # load video
    video = np.load(img_path, allow_pickle=True)
    video = video.swapaxes(0, 1)
    kpts_list = np.load(anno_path, allow_pickle=True)
    ef, edv, esv = kpts_list['ef'], kpts_list['edv'], kpts_list['esv']

    # Collect masks:
    idx_list = []
    masks = []
    mask_list = kpts_list['fnum_mask'].tolist()
    for kpt in kpts_list['fnum_mask'].tolist().keys():
        idx_list.append(int(kpt))
        masks.append(mask_list[kpt])

    # Swap if ED before ES:
    if idx_list[0] > idx_list[-1]:
        idx_list.reverse()
        masks.reverse()

    # compute step:
    x0, x1 = idx_list[-1], idx_list[0]
    if frame_length == -1:
        frame_length = x0 + 1
        
    step = min(x0, (x0 - x1) / (frame_length - 1))

    # select frames inds:
    frame_inds = [int(idx_list[0] + step * i) for i in range(frame_length)]

    # Collect frames:
    frames = []
    for i in range(frame_length):
        frames.append(video[frame_inds[i]])
        
    select_masks = []
    if len(masks) <= len(frame_inds):
        select_masks = masks
    else:
        for i in range(frame_length):
            select_masks.append(masks[frame_inds[i]])

    select_masks = np.asarray(select_masks)
    imgs = np.asarray(frames)

    spacing = kpts_list['spacing']

    return imgs, select_masks, ef, edv, esv, spacing


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)
    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


def random_click(mask, class_id=1):
    indices = np.argwhere(mask == class_id)
    indices[:, [0, 1]] = indices[:, [1, 0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0, 1]] = indices[:, [1, 0]]
    pt = indices[np.random.randint(len(indices))]
    return pt[np.newaxis, :], [point_label]


def fixed_click(mask, class_id=1):
    indices = np.argwhere(mask == class_id)
    indices[:, [0, 1]] = indices[:, [1, 0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0, 1]] = indices[:, [1, 0]]
    pt = indices[len(indices) // 2]
    return pt[np.newaxis, :], [point_label]


def random_clicks(mask, class_id=1, prompts_number=10):
    indices = np.argwhere(mask == class_id)
    indices[:, [0, 1]] = indices[:, [1, 0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0, 1]] = indices[:, [1, 0]]
    pt_index = np.random.randint(len(indices), size=prompts_number)
    pt = indices[pt_index]
    point_label = np.repeat(point_label, prompts_number)
    return pt, point_label


def pos_neg_clicks(mask, class_id=1, pos_prompt_number=5, neg_prompt_number=5):
    pos_indices = np.argwhere(mask == class_id)
    pos_indices[:, [0, 1]] = pos_indices[:, [1, 0]]
    pos_prompt_indices = np.random.randint(len(pos_indices),
                                           size=pos_prompt_number)
    pos_prompt = pos_indices[pos_prompt_indices]
    pos_label = np.repeat(1, pos_prompt_number)

    neg_indices = np.argwhere(mask != class_id)
    neg_indices[:, [0, 1]] = neg_indices[:, [1, 0]]
    neg_prompt_indices = np.random.randint(len(neg_indices),
                                           size=neg_prompt_number)
    neg_prompt = neg_indices[neg_prompt_indices]
    neg_label = np.repeat(0, neg_prompt_number)

    pt = np.vstack((pos_prompt, neg_prompt))
    point_label = np.hstack((pos_label, neg_label))
    return pt, point_label


def random_bbox(mask, class_id=1, img_size=256):
    # return box = np.array([x1, y1, x2, y2])
    indices = np.argwhere(mask == class_id)  # Y X
    indices[:, [0, 1]] = indices[:, [1, 0]]  # x, y
    if indices.shape[0] == 0:
        return np.array([-1, -1, img_size, img_size])

    # shiftw = randint(-int(0.9 * img_size), int(1.1 * img_size))
    # shifth = randint(-int(0.9 * img_size), int(1.1 * img_size))
    # shiftx = randint(-int(0.05 * img_size), int(0.05 * img_size))
    # shifty = randint(-int(0.05 * img_size), int(0.05 * img_size))

    minx = np.min(indices[:, 0])
    maxx = np.max(indices[:, 0])
    miny = np.min(indices[:, 1])
    maxy = np.max(indices[:, 1])

    classw_size = maxx-minx+1
    classh_size = maxy-miny+1

    shiftw = randint(int(0.95*classw_size), int(1.05*classw_size))
    shifth = randint(int(0.95*classh_size), int(1.05*classh_size))
    shiftx = randint(-int(0.05*classw_size), int(0.05*classw_size))
    shifty = randint(-int(0.05*classh_size), int(0.05*classh_size))

    new_centerx = (minx + maxx) // 2 + shiftx
    new_centery = (miny + maxy) // 2 + shifty

    minx = np.max([new_centerx - shiftw // 2, 0])
    maxx = np.min([new_centerx + shiftw // 2, img_size - 1])
    miny = np.max([new_centery - shifth // 2, 0])
    maxy = np.min([new_centery + shifth // 2, img_size - 1])

    return np.array([minx, miny, maxx, maxy])


def fixed_bbox(mask, class_id=1, img_size=256):
    indices = np.argwhere(mask == class_id)  # Y X (0, 1)
    indices[:, [0, 1]] = indices[:, [1, 0]]
    if indices.shape[0] == 0:
        return np.array([-1, -1, img_size, img_size])
    minx = np.min(indices[:, 0])
    maxx = np.max(indices[:, 0])
    miny = np.min(indices[:, 1])
    maxy = np.max(indices[:, 1])
    return np.array([minx, miny, maxx, maxy])

class JointTransform2D:
    """
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.

    Args:
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken.
        p_flip: float, the probability of performing a random horizontal flip.
        color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
            If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform using
            torchvision.transforms.RandomAffine.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    """

    def __init__(self,
                 img_size=256,
                 low_img_size=256,
                 ori_size=256,
                 crop=(32, 32),
                 p_flip=0.0,
                 p_rota=0.0,
                 p_scale=0.0,
                 p_gaussn=0.0,
                 p_contr=0.0,
                 p_gama=0.0,
                 p_distor=0.0,
                 color_jitter_params=(0.1, 0.1, 0.1, 0.1),
                 p_random_affine=0,
                 long_mask=False):
        self.crop = crop
        self.p_flip = p_flip
        self.p_rota = p_rota
        self.p_scale = p_scale
        self.p_gaussn = p_gaussn
        self.p_gama = p_gama
        self.p_contr = p_contr
        self.p_distortion = p_distor
        self.img_size = img_size
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask
        self.low_img_size = low_img_size
        self.ori_size = ori_size

    def __call__(self, image, mask):
        #  gamma enhancement
        if np.random.rand() < self.p_gama:
            c = 1
            g = np.random.randint(10, 25) / 10.0
            # g = 2
            image = (np.power(image / 255, 1.0 / g) / c) * 255
            image = image.astype(np.uint8)
        # transforming to PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)
        # random crop
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        # random horizontal flip
        if np.random.rand() < self.p_flip:
            image, mask = F.hflip(image), F.hflip(mask)
        # random rotation
        if np.random.rand() < self.p_rota:
            angle = T.RandomRotation.get_params((-30, 30))
            image, mask = F.rotate(image, angle), F.rotate(mask, angle)
        # random scale and center resize to the original size
        if np.random.rand() < self.p_scale:
            scale = np.random.uniform(1, 1.3)
            new_h, new_w = int(self.img_size * scale), int(self.img_size *
                                                           scale)
            image, mask = F.resize(image, (new_h, new_w),
                                   InterpolationMode.BILINEAR), F.resize(
                                       mask, (new_h, new_w),
                                       InterpolationMode.NEAREST)
            # image = F.center_crop(image, (self.img_size, self.img_size))
            # mask = F.center_crop(mask, (self.img_size, self.img_size))
            i, j, h, w = T.RandomCrop.get_params(
                image, (self.img_size, self.img_size))
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        # random add gaussian noise
        if np.random.rand() < self.p_gaussn:
            ns = np.random.randint(3, 15)
            noise = np.random.normal(
                loc=0, scale=1, size=(self.img_size, self.img_size)) * ns
            noise = noise.astype(int)
            image = np.array(image) + noise
            image[image > 255] = 255
            image[image < 0] = 0
            image = F.to_pil_image(image.astype('uint8'))
        # random change the contrast
        if np.random.rand() < self.p_contr:
            contr_tf = T.ColorJitter(contrast=(0.8, 2.0))
            image = contr_tf(image)
        # random distortion
        if np.random.rand() < self.p_distortion:
            distortion = T.RandomAffine(0, None, None, (5, 30))
            image = distortion(image)
        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)
        # random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params(
                (-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            image, mask = F.affine(image, *affine_params), F.affine(
                mask, *affine_params)
        # transforming to tensor
        image, mask = F.resize(image, (self.img_size, self.img_size),
                               InterpolationMode.BILINEAR), F.resize(
                                   mask, (self.ori_size, self.ori_size),
                                   InterpolationMode.NEAREST)
        low_mask = F.resize(mask, (self.low_img_size, self.low_img_size),
                            InterpolationMode.NEAREST)
        image = F.to_tensor(image)

        if not self.long_mask:
            mask = F.to_tensor(mask)
            low_mask = F.to_tensor(low_mask)
        else:
            mask = to_long_tensor(mask)
            low_mask = to_long_tensor(low_mask)
        return image, mask, low_mask


class JointTransform3D:
    """
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.

    Args:
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken.
        p_flip: float, the probability of performing a random horizontal flip.
        color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
            If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform using
            torchvision.transforms.RandomAffine.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    """

    def __init__(self,
                 img_size=256,
                 low_img_size=256,
                 ori_size=256,
                 crop=(32, 32),
                 p_flip=0.0,
                 p_rota=0.0,
                 p_scale=0.0,
                 p_gaussn=0.0,
                 p_contr=0.0,
                 p_gama=0.0,
                 p_distor=0.0,
                 color_jitter_params=(0.1, 0.1, 0.1, 0.1),
                 p_random_affine=0,
                 long_mask=False):
        self.crop = crop
        self.p_flip = p_flip
        self.p_rota = p_rota
        self.p_scale = p_scale
        self.p_gaussn = p_gaussn
        self.p_gama = p_gama
        self.p_contr = p_contr
        self.p_distortion = p_distor
        self.img_size = img_size
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask
        self.low_img_size = low_img_size
        self.ori_size = ori_size

    def __call__(self, image, mask):
        image = image.astype(np.uint8)
        #  gamma enhancement
        if np.random.rand() < self.p_gama:
            c = 1
            g = np.random.randint(10, 25) / 10.0
            # g = 2
            image = (np.power(image / 255, 1.0 / g) / c) * 255
            image = image.astype(np.uint8)
        # transforming to PIL image
        image_list, mask_list = [], []
        for image_ in image:
            image_ = image_.transpose(1,2,0)
            image_ = F.to_pil_image(image_)
            image_list.append(image_)
        for mask_ in mask:
            mask_ = F.to_pil_image(mask_)
            mask_list.append(mask_)

        # image, mask = F.to_pil_image(image), F.to_pil_image(mask)
        # image = F.to_pil_image(image)
        # random crop
        if self.crop: # [112,112]
            i, j, h, w = T.RandomCrop.get_params(image_list[0], self.crop)
            for idx, image_ in enumerate(image_list):
                image_list[idx] = F.crop(image_, i, j, h, w)
            for idx, mask_ in enumerate(mask_list):
                mask_list[idx] = F.crop(mask_, i, j, h, w)
        # random horizontal flip
        if np.random.rand() < self.p_flip:
            image_list = [F.hflip(image_) for image_ in image_list]
            mask_list = [F.hflip(mask_) for mask_ in mask_list]
        # random rotation
        if np.random.rand() < self.p_rota:
            angle = T.RandomRotation.get_params((-30, 30))
            image_list = [F.rotate(image_, angle) for image_ in image_list]
            mask_list = [F.rotate(mask_, angle) for mask_ in mask_list]
        # random scale and center resize to the original size
        if np.random.rand() < self.p_scale:
            scale = np.random.uniform(1, 1.3)
            new_h, new_w = int(self.img_size * scale), int(self.img_size * scale)
            image_list = [
                F.resize(image_, (new_h, new_w), InterpolationMode.BILINEAR)
                for image_ in image_list
            ]
            mask_list = [
                F.resize(mask_, (new_h, new_w), InterpolationMode.NEAREST)
                for mask_ in mask_list
            ]
            i, j, h, w = T.RandomCrop.get_params(image_list[0], (self.img_size, self.img_size))
            for idx, image_ in enumerate(image_list):
                image_list[idx] = F.crop(image_, i, j, h, w)
            for idx, mask_ in enumerate(mask_list):
                mask_list[idx] = F.crop(mask_, i, j, h, w)

        # random add gaussian noise
        # if np.random.rand() < self.p_gaussn:
        #     ns = np.random.randint(3, 15)
        #     noise = np.random.normal(
        #         loc=0, scale=1, size=(self.img_size, self.img_size)) * ns
        #     noise = noise.astype(int)
        #     for idx, image_ in enumerate(image_list):
        #         image_ = np.array(image_) + noise # BUG
        #         image_[image_ > 255] = 255
        #         image_[image_ < 0] = 0
        #         image_ = F.to_pil_image(image_.astype('uint8'))
        #         image_list[idx] = image_

        # random change the contrast
        if np.random.rand() < self.p_contr:
            contr_tf = T.ColorJitter(contrast=(0.8, 2.0))
            image_list = [contr_tf(image_) for image_ in image_list]

        # random distortion
        # if np.random.rand() < self.p_distortion: # BUG?
        #     distortion = T.RandomAffine(0, None, None, (5, 30))
        #     image_list = [distortion(image_) for image_ in image_list]

        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)

        # random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params(
                (-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            image_list = [F.affine(image_, *affine_params) for image_ in image_list]
            mask_list = [F.affine(mask_, *affine_params) for mask_ in mask_list]

        # transforming to tensor
        image_list = [
            F.resize(image_, (self.img_size, self.img_size), InterpolationMode.BILINEAR)
            for image_ in image_list
        ]
        mask_list = [
            F.resize(mask_, (self.img_size, self.img_size), InterpolationMode.NEAREST)
            for mask_ in mask_list
        ]
        image = np.stack(image_list)
        image = image.transpose(0,3,1,2)

        mask = np.stack(mask_list)
        
        image = torch.tensor(image)
        mask = torch.tensor(mask)

        # image = F.to_tensor(image)

        # if not self.long_mask:
        #     mask = F.to_tensor(mask)
        #     low_mask = F.to_tensor(low_mask)
        # else:
        #     mask = to_long_tensor(mask)
        #     low_mask = to_long_tensor(low_mask)
        return image, mask


class ImageToImage2D(Dataset):
    """
    Reads the images and applies the augmentation transform on them.

    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
                |-- MainPatient
                    |-- train.txt
                    |-- val.txt
                    |-- text.txt 
                        {subtaski}/{imgname}
                    |-- class.json
                |-- subtask1
                    |-- img
                        |-- img001.png
                        |-- img002.png
                        |-- ...
                    |-- label
                        |-- img001.png
                        |-- img002.png
                        |-- ...
                |-- subtask2
                    |-- img
                        |-- img001.png
                        |-- img002.png
                        |-- ...
                    |-- label
                        |-- img001.png
                        |-- img002.png
                        |-- ... 
                |-- subtask...   

        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """

    def __init__(self,
                 dataset_path: str,
                 split='train',
                 joint_transform: Callable = None,
                 img_size=256,
                 prompt="click",
                 class_id=1,
                 one_hot_mask: int = False) -> None:
        self.dataset_path = dataset_path
        self.one_hot_mask = one_hot_mask
        self.split = split
        id_list_file = os.path.join(dataset_path,
                                    'MainPatient/{0}.txt'.format(split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.prompt = prompt
        self.img_size = img_size
        self.class_id = class_id
        self.class_dict_file = os.path.join(dataset_path,
                                            'MainPatient/class.json')
        with open(self.class_dict_file, 'r') as load_f:
            self.class_dict = json.load(load_f)
        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        if "test" in self.split:
            sub_path, filename = id_.split('/')[0], id_.split('/')[1]
            # class_id0, sub_path, filename = id_.split('/')[0], id_.split('/')[1], id_.split('/')[2]
            # self.class_id = int(class_id0)
        else:
            class_id0, sub_path, filename = id_.split('/')[0], id_.split(
                '/')[1], id_.split('/')[2]
        img_path = os.path.join(os.path.join(self.dataset_path, sub_path),
                                'img')
        label_path = os.path.join(os.path.join(self.dataset_path, sub_path),
                                  'label')
        image = cv2.imread(os.path.join(img_path, filename + '.png'), 0)
        mask = cv2.imread(os.path.join(label_path, filename + '.png'), 0)
        classes = self.class_dict[sub_path]
        if classes == 2:
            mask[mask > 1] = 1

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
        if self.joint_transform:
            image, mask, low_mask = self.joint_transform(image, mask)
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1],
                                mask.shape[2])).scatter_(0, mask.long(), 1)

        # --------- make the point prompt -----------------
        if self.prompt == 'click':
            point_label = 1
            if 'train' in self.split:
                #class_id = randint(1, classes-1)
                class_id = int(class_id0)
            elif 'val' in self.split:
                class_id = int(class_id0)
            else:
                class_id = self.class_id
            if 'train' in self.split:
                pt, point_label = random_click(np.array(mask), class_id)
                bbox = random_bbox(np.array(mask), class_id, self.img_size)
            else:
                pt, point_label = fixed_click(np.array(mask), class_id)
                bbox = fixed_bbox(np.array(mask), class_id, self.img_size)
            mask[mask != class_id] = 0
            mask[mask == class_id] = 1
            low_mask[low_mask != class_id] = 0
            low_mask[low_mask == class_id] = 1
            point_labels = np.array(point_label)
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1],
                                mask.shape[2])).scatter_(0, mask.long(), 1)

        low_mask = low_mask.unsqueeze(0)
        mask = mask.unsqueeze(0)
        return {
            'image': image,
            'label': mask,
            'p_label': point_labels,
            'pt': pt,
            'bbox': bbox,
            'low_mask': low_mask,
            'image_name': filename + '.png',
            'class_id': class_id,
        }


class EchoDataset(Dataset):
    """
    Reads the images and applies the augmentation transform on them.

    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
                |-- MainPatient
                    |-- train.txt
                    |-- val.txt
                    |-- text.txt 
                        {subtaski}/{imgname}
                    |-- class.json
                |-- subtask1
                    |-- img
                        |-- img001.png
                        |-- img002.png
                        |-- ...
                    |-- label
                        |-- img001.png
                        |-- img002.png
                        |-- ...
                |-- subtask2
                    |-- img
                        |-- img001.png
                        |-- img002.png
                        |-- ...
                    |-- label
                        |-- img001.png
                        |-- img002.png
                        |-- ... 
                |-- subtask...   

        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """

    def __init__(self,
                 dataset_path: str,
                 split='train',
                 joint_transform: Callable = None,
                 img_size=256,
                 prompt="click",
                 class_id=1,
                 one_hot_mask: int = False) -> None:
        self.dataset_path = dataset_path
        self.one_hot_mask = one_hot_mask
        self.split = split
        id_list_file = os.path.join(dataset_path, '{0}.txt'.format(split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.prompt = prompt
        self.img_size = img_size
        self.class_id = class_id
        self.class_dict_file = os.path.join(dataset_path, 'class.json')
        with open(self.class_dict_file, 'r') as load_f:
            self.class_dict = json.load(load_f)
        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        filename = id_
        sub_path = 'EchoNet'
        class_id0 = 1
        split = self.split.split('_')[1]

        img_path = os.path.join(os.path.join(self.dataset_path, 'images'),
                                split)
        label_path = os.path.join(
            os.path.join(self.dataset_path, 'annotations'), split)
        image = cv2.imread(os.path.join(img_path, filename), 0)
        mask = cv2.imread(os.path.join(label_path, filename), 0)
        classes = self.class_dict[sub_path]
        if classes == 2:
            mask[mask > 1] = 1

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
        if self.joint_transform:
            image, mask, low_mask = self.joint_transform(image, mask)
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1],
                                mask.shape[2])).scatter_(0, mask.long(), 1)

        # --------- make the point prompt -----------------
        if self.prompt == 'click':
            point_label = 1
            if 'train' in self.split:
                #class_id = randint(1, classes-1)
                class_id = int(class_id0)
            elif 'val' in self.split:
                class_id = int(class_id0)
            else:
                class_id = self.class_id
            if 'train' in self.split:
                pt, point_label = random_click(np.array(mask), class_id)
                bbox = random_bbox(np.array(mask), class_id, self.img_size)
            else:
                pt, point_label = fixed_click(np.array(mask), class_id)
                bbox = fixed_bbox(np.array(mask), class_id, self.img_size)
            mask[mask != class_id] = 0
            mask[mask == class_id] = 1
            low_mask[low_mask != class_id] = 0
            low_mask[low_mask == class_id] = 1
            point_labels = np.array(point_label)
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1],
                                mask.shape[2])).scatter_(0, mask.long(), 1)

        low_mask = low_mask.unsqueeze(0)
        mask = mask.unsqueeze(0)
        return {
            'image': image,
            'label': mask,
            'p_label': point_labels,
            'pt': pt,
            'bbox': bbox,
            'low_mask': low_mask,
            'image_name': filename,
            'class_id': class_id,
        }


class EchoVideoDataset(Dataset):

    def __init__(self,
                 dataset_path: str,
                 split='train',
                 joint_transform: Callable = None,
                 img_size=256,
                 prompt="click",
                 class_id=1,
                 one_hot_mask: int = False,
                 frame_length: int = 2,
                 disable_point_prompt: bool = True,
                 point_numbers: int = 1) -> None:
        self.dataset_path = dataset_path
        self.one_hot_mask = one_hot_mask
        self.split = split
        self.frame_length = frame_length
        self.point_numbers = point_numbers
        self.ids = []
        for _, _, files in os.walk(os.path.join(dataset_path, 'videos',
                                                split)):
            self.ids = files

        # id_list_file = os.path.join(dataset_path, '{0}.txt'.format(split))
        # self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.prompt = prompt
        self.disable_point_prompt = disable_point_prompt
        self.img_size = img_size
        self.class_id = class_id
        self.class_dict_file = os.path.join(dataset_path, 'class.json')
        with open(self.class_dict_file, 'r') as load_f:
            self.class_dict = json.load(load_f)
        if joint_transform:
            self.joint_transform = joint_transform
        else:
            # to_tensor = T.ToTensor()
            to_tensor = torch.tensor
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        filename = self.ids[i]
        prefix, _ = os.path.splitext(filename)
        sub_path = 'EchoNet'
        class_id = 1

        img_path = os.path.join(os.path.join(self.dataset_path, 'videos'), self.split)
        label_path = os.path.join(os.path.join(self.dataset_path, 'annotations'), self.split)
        image, mask, ef, edv, esv, spacing = load_video_and_mask_file(
            img_path=os.path.join(img_path, prefix + '.npy'),
            anno_path=os.path.join(label_path, prefix + '.npz'),
            frame_length=self.frame_length)
        classes = self.class_dict[sub_path]
        if classes == 2:
            mask[mask > 1] = 0

        # data aug
        # correct dimensions if needed
        # image, mask = correct_dims(image, mask)
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)
            
        # --------- make the point prompt -----------------
        pts, point_labels = [], []  
        pt = []
        if not self.disable_point_prompt:
            if self.prompt == 'click':
                if 'train' in self.split:
                    for mask_ in mask:
                        pt, point_label = random_click(np.array(mask_), class_id)
                        if self.point_numbers > 1:
                            for i in range(1, self.point_numbers):
                                _pt, _point_label = random_click(np.array(mask_), class_id)
                                pt = np.concatenate([pt, _pt], axis=0)
                                point_label = np.concatenate([point_label, _point_label], axis=0)
                        pts.append(pt)
                        point_labels.append(point_label)
                else:
                    for mask_ in mask:
                        pt, point_label = fixed_click(np.array(mask_), class_id)
                        if self.point_numbers > 1:
                            for i in range(1, self.point_numbers):
                                _pt, _point_label = fixed_click(np.array(mask_), class_id)
                                pt = np.concatenate([pt, _pt], axis=0)
                                point_label = np.concatenate([point_label, _point_label], axis=0)
                        pts.append(pt)
                        point_labels.append(point_label)
                pt = np.stack(pts)
                point_label = np.stack(point_labels)
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1],
                                mask.shape[2])).scatter_(0, mask.long(), 1)

        # low_mask = low_mask.unsqueeze(0)

        return {
            'image': image,
            'label': mask,
            'p_label': point_labels,
            'pt': pt,
            # 'low_mask': low_mask,
            'image_name': filename,
            'class_id': class_id,
            'ef': ef,
            'edv': edv,
            'esv': esv,
            'spacing':spacing,
        }


class Logger:

    def __init__(self, verbose=False):
        self.logs = defaultdict(list)
        self.verbose = verbose

    def log(self, logs):
        for key, value in logs.items():
            self.logs[key].append(value)

        if self.verbose:
            print(logs)

    def get_logs(self):
        return self.logs

    def to_csv(self, path):
        pd.DataFrame(self.logs).to_csv(path, index=None)
