# MemSAM
[**MemSAM: Taming Segment Anything Model for Echocardiography Video Segmentation**](https://openaccess.thecvf.com/content/CVPR2024/papers/Deng_MemSAM_Taming_Segment_Anything_Model_for_Echocardiography_Video_Segmentation_CVPR_2024_paper.pdf), CVPR 2024, _Oral_

Xiaolong Deng^, [Huisi Wu*](https://csse.szu.edu.cn/staff/~hswu/), [Runhao Zeng](https://zengrunhao.com/), [Jing Qin](https://research.polyu.edu.hk/en/persons/jing-qin)

[[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Deng_MemSAM_Taming_Segment_Anything_Model_for_Echocardiography_Video_Segmentation_CVPR_2024_paper.pdf) [[Video]](https://www.youtube.com/watch?v=N2usOkkNHQs) [[Project]](https://github.com/dengxl0520/MemSAM)
<!-- ![MemSAM Design](/assets/framework.jpg) -->

<div align=center>
<img src="/assets/framework.jpg" width="600" alt="MemSAM Design" />
</div>

<!-- The code will be uploaded later. -->

## Installation
```
conda create --name memsam python=3.10
conda activate memsam
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install requirements.txt
```

## Usage
### prepare dataset
First, download the dataset from:
- [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/index.html)
- [EchoNet-Dynamic](https://echonet.github.io/dynamic/index.html)
  
Then process the dataset according to `utils/preprocess_echonet.py` and `utils/preprocess_camus.py`, for example:

```
# CAMUS
python utils/preprocess_camus.py -i /data/dengxiaolong/CAMUS_public/database_nifti -o /data/dengxiaolong/memsam/CAMUS_public

# EchoNet-Dynamic
python utils/preprocess_echonet.py -i /data/dengxiaolong/EchoNet-Dynamic -o /data/dengxiaolong/memsam/EchoNet
```

### pretrain checkpoint download
[ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

### train and test
Use `train_video.py` and `test_video.py` to train and test separately.

## Acknowledgement
The work is based on [SAM](https://github.com/facebookresearch/segment-anything), [SAMUS](https://github.com/xianlin7/SAMUS) and [XMem](https://github.com/hkchengrex/XMem). Thanks for the open source contributions to these efforts!

## Citation
if you find our work useful, please cite our paper, thank you!
```
@InProceedings{Deng_2024_CVPR,
    author    = {Deng, Xiaolong and Wu, Huisi and Zeng, Runhao and Qin, Jing},
    title     = {MemSAM: Taming Segment Anything Model for Echocardiography Video Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {9622-9631}
}
```
