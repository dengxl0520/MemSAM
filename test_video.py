from ast import arg
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
from pickle import FALSE, TRUE
from statistics import mode
from tkinter import image_names
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import random
from utils.config import get_config
from utils.evaluation import get_eval
from importlib import import_module

from torch.nn.modules.loss import CrossEntropyLoss
from monai.losses import DiceCELoss
from einops import rearrange
from models.model_dict import get_model
from utils.data_us import EchoVideoDataset, JointTransform3D, EchoDataset
from utils.loss_functions.sam_loss import get_criterion
from utils.generate_prompts import get_click_prompt
from easydict import EasyDict


args = EasyDict(base_lr=0.0005,
            batch_size=1,
            encoder_input_size=256,
            keep_log=False,
            low_image_size=256,
            modelname='MemSAM',
            n_gpu=1,
            sam_ckpt='checkpoints/sam_vit_b_01ec64.pth',
            task='CAMUS_Video_Full',
            vit_name='vit_b',
            enable_memory=True,
            disable_point_prompt=False,
            point_numbers_prompt=True,
            point_numbers=1,
            reinforce=True,
            compute_ef=True)

opt = get_config(args.task)  # please configure your hyper-parameter
opt.load_path = 'checkpoints/CAMUS_full/your_checkpoint.pth'
print("task", args.task, "checkpoints:", opt.load_path)
opt.mode = "test"
#opt.classes=2
opt.semi = False
opt.visual = True
#opt.eval_mode = "patient"
opt.modelname = 'MemSAM'
device = torch.device(opt.device)

# ==================================================set random seed==================================================
seed_value = 1234  # the number of seed
np.random.seed(seed_value)  # set random seed for numpy
random.seed(seed_value)  # set random seed for python
os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
torch.manual_seed(seed_value)  # set random seed for CPU
torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
torch.backends.cudnn.deterministic = True  # set random seed for convolution
torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)

opt.batch_size = args.batch_size * args.n_gpu

tf_val = JointTransform3D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
test_dataset = EchoVideoDataset(opt.data_path, opt.test_split,  tf_val, img_size=args.encoder_input_size, frame_length=10,disable_point_prompt=args.disable_point_prompt, point_numbers=args.point_numbers)  # return image, mask, and filename
testloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0, pin_memory=True)

model = get_model(args.modelname, args=args, opt=opt)
model.to(device)
model.train()

checkpoint = torch.load(opt.load_path)
#------when the load model is saved under multiple GPU
new_state_dict = {}
for k,v in checkpoint.items():
    if k[:7] == 'module.':
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v
model.load_state_dict(new_state_dict)

optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
#optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
criterion = get_criterion(modelname=args.modelname, opt=opt)

#  ========================================================================= begin to evaluate the model ============================================================================

# pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("Total_params: {}".format(pytorch_total_params))
input = torch.randn(1, 1, 3, args.encoder_input_size, args.encoder_input_size).cuda()
points = (torch.tensor([[[[1, 2]]]]).float().cuda(), torch.tensor([[1]]).float().cuda())
from thop import profile
flops, params = profile(model, inputs=(input, points), )
print('Gflops:', flops/1000000000, 'params:', params)

# sum_time = 0
# with torch.no_grad():
#     start_time = time.time()
#     pred = model(input, points, None)
#     sum_time =  sum_time + (time.time()-start_time)
# print("test speed", sum_time)

model.eval()
dice_mean, iou_mean, hd_mean, assd_mean, dices_std, iou_std, hd_std, assd_std = get_eval(testloader, model, criterion=criterion, opt=opt, args=args)
print("dataset:" + args.task + " -----------model name: "+ args.modelname)
print("task", args.task, "checkpoints:", opt.load_path)
print("dice_mean  iou_mean  hd_mean  assd_mean")
print(dice_mean, iou_mean, hd_mean, assd_mean)
print("dices_std  iou_std  hd_std  assd_std")
print(dices_std, iou_std, hd_std, assd_std)

