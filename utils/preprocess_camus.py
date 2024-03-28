import argparse
import os
import json
import SimpleITK as sitk
import cv2
import numpy as np
import random
from PIL import Image
from compute_ef import *
from pathlib import Path

SEED = 42
RESIZE_SIZE = (256,256)
SPLIT_RATIOS = [0.7,0.1,0.2]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='/data/dengxiaolong/CAMUS_public/database_nifti')
    parser.add_argument('-o', '--output_dir', type=str, default='/tmp/CAMUS_public_256')
    parser.add_argument('-f', '--split_file', type=str)
    args = parser.parse_args()

    return args

def sitk_load_resize(filepath: str | Path, resize_size) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Loads an image using SimpleITK and returns the image and its metadata.

    Args:
        filepath: Path to the image.

    Returns:
        - ([N], H, W), Image array.
        - Collection of metadata.
    """
    # Load image and save info
    image = sitk.ReadImage(str(filepath))
    image = resampleXYSize(image, *resize_size)
    info = {"origin": image.GetOrigin(), "spacing": image.GetSpacing(), "direction": image.GetDirection()}

    # Extract numpy array from the SimpleITK image object
    im_array = np.squeeze(sitk.GetArrayFromImage(image))

    return im_array, info

def generate_list(begin=1, end=450):
    number_list = ['patient'+str(i).zfill(4) for i in range(begin,end+1)]
    return number_list

def read_cfg(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    data = {}

    for line in lines:
        line = line.strip()
        key, value = line.split(":")
        key = key.strip()
        value = value.strip()
        data[key] = value

    return data

def random_split(data:list, ratios:list):
    total_length = len(data)
    split_points = [int(ratio * total_length) for ratio in ratios]
    random.seed(SEED)
    random.shuffle(data)
    splits = []
    start = 0
    for point in split_points:
        splits.append(data[start:start+point])
        start += point
    return splits

def split(data:list, ratios:list):
    total_length = len(data)
    split_points = [int(ratio * total_length) for ratio in ratios]
    splits = []
    start = 0
    for point in split_points:
        splits.append(data[start:start+point])
        start += point
    return splits

def filter(x):
    # remove error value
    x = np.where(x != 0, 255, 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(x, connectivity=8)
    max_area_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    x = np.where(labels == max_area_label, 1, 0).astype(np.uint8)
    return x

# from https://aistudio.baidu.com/projectdetail/1915947
def resampleSpacing(sitkImage, newspace=(1,1,1)):
    '''
        newResample = resampleSpacing(sitkImage, newspace=[1,1,1])
    '''
    euler3d = sitk.Euler3DTransform()
    xsize, ysize, zsize = sitkImage.GetSize()
    xspacing, yspacing, zspacing = sitkImage.GetSpacing()
    origin = sitkImage.GetOrigin()
    direction = sitkImage.GetDirection()
    #新的X轴的Size = 旧X轴的Size *（原X轴的Spacing / 新设定的Spacing）
    new_size = (int(xsize*xspacing/newspace[0]),int(ysize*yspacing/newspace[1]),int(zsize*zspacing/newspace[2]))
    #如果是对标签进行重采样，模式使用最近邻插值，避免增加不必要的像素值
    sitkImage = sitk.Resample(sitkImage,new_size,euler3d,sitk.sitkNearestNeighbor,origin,newspace,direction)
    return sitkImage

def resampleSize(sitkImage, depth):
    '''
        newsitkImage = resampleSize(sitkImage, depth=DEPTH)
    '''
    #重采样函数
    euler3d = sitk.Euler3DTransform()

    xsize, ysize, zsize = sitkImage.GetSize()
    xspacing, yspacing, zspacing = sitkImage.GetSpacing()
    new_spacing_z = zspacing/(depth/float(zsize))

    origin = sitkImage.GetOrigin()
    direction = sitkImage.GetDirection()
    #根据新的spacing 计算新的size
    newsize = (xsize,ysize,int(zsize*zspacing/new_spacing_z))
    newspace = (xspacing, yspacing, new_spacing_z)
    sitkImage = sitk.Resample(sitkImage,newsize,euler3d,sitk.sitkNearestNeighbor,origin,newspace,direction)
    return sitkImage

def resampleXYSize(sitkImage, new_xsize, new_ysize):
    '''
        newsitkImage = resampleSize(sitkImage, depth=DEPTH)
    '''
    #重采样函数
    euler3d = sitk.Euler3DTransform()

    xsize, ysize, zsize = sitkImage.GetSize()
    xspacing, yspacing, zspacing = sitkImage.GetSpacing()
    new_spacing_x = xspacing/(new_xsize/float(xsize))
    new_spacing_y = yspacing/(new_ysize/float(ysize))

    origin = sitkImage.GetOrigin()
    direction = sitkImage.GetDirection()
    #根据新的spacing 计算新的size
    newsize = (new_xsize,new_ysize,zsize)
    newspace = (new_spacing_x, new_spacing_y, zspacing)
    sitkImage = sitk.Resample(sitkImage,newsize,euler3d,sitk.sitkNearestNeighbor,origin,newspace,direction)
    return sitkImage

def compute_ef_to_patient(patient_dir: Path):
    patient_name = patient_dir.name 
    gt_mask_pattern = "{patient_name}_{view}_{instant}_gt.nii.gz"
    lv_label = 1

    view = "2CH"
    instant = "ED"
    a2c_ed, a2c_info = sitk_load(patient_dir / gt_mask_pattern.format(patient_name=patient_name, view=view, instant=instant))
    a2c_voxelspacing = a2c_info["spacing"][:2][::-1]    # Extract the (width,height) dimension from the metadata and order them like in the mask

    instant = "ES"
    a2c_es, _ = sitk_load(patient_dir / gt_mask_pattern.format(patient_name=patient_name, view=view, instant=instant))

    view = "4CH"
    instant = "ED"
    a4c_ed, a4c_info = sitk_load(patient_dir / gt_mask_pattern.format(patient_name=patient_name, view=view, instant=instant))
    a4c_voxelspacing = a4c_info["spacing"][:2][::-1]    # Extract the (width,height) dimension from the metadata and order them like in the mask

    instant = "ES"
    a4c_es, _ = sitk_load(patient_dir / gt_mask_pattern.format(patient_name=patient_name, view=view, instant=instant))
    # Extract binary LV masks from the multi-class segmentation masks
    a2c_ed_lv_mask = a2c_ed == lv_label
    a2c_es_lv_mask = a2c_es == lv_label
    a4c_ed_lv_mask = a4c_ed == lv_label
    a4c_es_lv_mask = a4c_es == lv_label

    # Use the provided implementation to compute the LV volumes
    edv, esv = compute_left_ventricle_volumes(a2c_ed_lv_mask, a2c_es_lv_mask, a2c_voxelspacing, a4c_ed_lv_mask, a4c_es_lv_mask, a4c_voxelspacing)
    ef = round(100 * (edv - esv) / edv, 2) # Round the computed value to the nearest integer

    # print(f"{patient_name=}: {ef=}, {edv=}, {esv=}")
    return ef, edv, esv

def preprocess_data(input_path, output_path, split_file):
    # hyperparam
    resize_size = RESIZE_SIZE
    # generate video name
    patient_name_list = generate_list(1,500)

    # split dataset
    if split_file:
        # from split file
        with open(split_file, 'r') as f:
            data = json.load(f)
            train_split, val_split, test_split = data['train'],data['val'],data['test']
    else:
        # random split
        data_split = split(data=patient_name_list, ratios=SPLIT_RATIOS)
        train_split, val_split, test_split = data_split

    # create folder
    if not os.path.exists(output_path + '/videos'):
        os.makedirs(output_path + '/videos/train')
        os.makedirs(output_path + '/videos/val')
        os.makedirs(output_path + '/videos/test')
    if not os.path.exists(output_path + '/annotations'):
        os.makedirs(output_path + '/annotations/train')
        os.makedirs(output_path + '/annotations/val')
        os.makedirs(output_path + '/annotations/test')

    for idx, patient_name in enumerate(patient_name_list):
        patient_root = Path(input_path)
        patient_dir = patient_root / patient_name

        seq_pattern = "{patient_name}_{view}_half_sequence.nii.gz"
        seq_gt_pattern = "{patient_name}_{view}_half_sequence_gt.nii.gz"
        cfg_pattern = "Info_{view}.cfg"
        
        seq_save_pattern = "{patient_name}_{view}.npy"
        seq_gt_save_pattern = "{patient_name}_{view}.npz"

        ef, edv, esv = compute_ef_to_patient(patient_dir=patient_dir)

        for view in ['2CH','4CH']:
            seq_name = seq_pattern.format(patient_name=patient_name, view=view)
            seq_gt_name = seq_gt_pattern.format(patient_name=patient_name, view=view)

            seq, seq_info = sitk_load_resize(patient_dir / seq_name, resize_size)
            seq_gt, seq_gt_info = sitk_load_resize(patient_dir / seq_gt_name, resize_size)

            assert seq_info['spacing'] == seq_gt_info['spacing']

            cfg = read_cfg(patient_dir / cfg_pattern.format(view=view))
            assert cfg['ED'] == cfg['NbFrame'] or cfg["ES"] == cfg['NbFrame']
            # let ed -> es
            if int(cfg['ED']) > int(cfg['ES']) :
                # print(patient_dir,view,':es -> ed')
                # flip video
                seq, seq_gt= np.flip(seq, axis=0), np.flip(seq_gt, axis=0)
                cfg['ED'], cfg['ES'] = cfg['ES'], cfg['ED']
                    
            # to rgb
            seq = np.repeat(seq[np.newaxis, :, :, :], 3, axis=0)

            frame_pairs_mask = {}
            for idxx, frame_gt in enumerate(seq_gt):
                frame_pairs_mask[str(idxx)] = frame_gt

            # save
            if patient_name in train_split:
                video_save_path = os.path.join(output_path, 'videos/train')
                anno_save_path = os.path.join(output_path, 'annotations/train')
            if patient_name in val_split:
                video_save_path = os.path.join(output_path, 'videos/val')
                anno_save_path = os.path.join(output_path, 'annotations/val')
            if patient_name in test_split:
                video_save_path = os.path.join(output_path, 'videos/test')
                anno_save_path = os.path.join(output_path, 'annotations/test')

            # save video
            np.save(os.path.join(video_save_path, seq_save_pattern.format(patient_name=patient_name, view=view)), seq)
            # save anno
            np.savez(
                os.path.join(anno_save_path, seq_gt_save_pattern.format(patient_name=patient_name, view=view)),
                fnum_mask=frame_pairs_mask,
                ef=ef,
                edv=edv,
                esv=esv,
                spacing=seq_info['spacing']
            )
            print(idx+1, seq_save_pattern.format(patient_name=patient_name, view=view))
    
    # creaet split txt
    output_file_train = open(output_path + '/camus_train_filenames.txt', 'w')
    output_file_val = open(output_path + '/camus_val_filenames.txt', 'w')
    output_file_test = open(output_path + '/camus_test_filenames.txt', 'w')

    for name in train_split:
        output_file_train.write(name + '\n')
    for name in val_split:
        output_file_val.write(name + '\n')
    for name in test_split:
        output_file_test.write(name + '\n')

if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.input_dir):
        raise ValueError('Input directory does not exist.')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    preprocess_data(input_path=args.input_dir,
                    output_path=args.output_dir,
                    split_file=args.split_file)
