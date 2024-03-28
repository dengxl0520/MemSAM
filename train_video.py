import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
from pickle import FALSE, TRUE
from statistics import mode
from tkinter import image_names
from easydict import EasyDict
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
from utils.data_us import EchoVideoDataset, JointTransform3D
from utils.data_us import JointTransform2D, EchoDataset
from utils.loss_functions.sam_loss import get_criterion
from utils.generate_prompts import get_click_prompt


def main():

    #  ============================================================================= parameters setting ====================================================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='XMemSAM', type=str, help='type of model, e.g., SAM, SAMFull, MedSAM, MSA, SAMed, SAMUS...')
    parser.add_argument('--encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS')
    parser.add_argument('--low_image_size', type=int, default=256, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS')
    parser.add_argument('--task', default='CAMUS_Video_Full', help='task or dataset name: CAMUS_Video_Full or EchoNet_Video')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu') # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA') #0.0006
    parser.add_argument('--warmup', action="store_true", help='If activated, warp up the learning from a lower lr to the base_lr') 
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('--keep_log', action="store_true", help='keep the loss&lr&dice during training or not')
    parser.add_argument('--frame_length', type=int, default=10)
    parser.add_argument('--point_numbers', type=int, default=1)
    parser.add_argument('--enable_memory', action="store_true")
    parser.add_argument('--semi', action="store_true")
    parser.add_argument('--reinforce', action="store_true")
    parser.add_argument('--disable_point_prompt', action="store_true")
    args = parser.parse_args()
    print(args)

    # ==================================================parameters setting==================================================

    # override_args = EasyDict(base_lr=0.0001,
    #                     batch_size=1,
    #                     encoder_input_size=256,
    #                     keep_log=True,
    #                     low_image_size=256,
    #                     frame_length=10,
    #                     modelname='XMemSAM',
    #                     n_gpu=1,
    #                     sam_ckpt='checkpoints/sam_vit_b_01ec64.pth',
    #                     task='CAMUS_Video_Full',
    #                     vit_name='vit_b',
    #                     enable_memory=True,
    #                     enable_point_prompt=True,
    #                     point_numbers=1,
    #                     warmup=False,
    #                     warmup_period=250)
    opt = get_config(args.task)
    opt.semi = args.semi

    device = torch.device(opt.device)
    if args.keep_log:
        logtimestr = time.strftime(
            '%m%d%H%M'
        )  # initialize the tensorboard for record the training process
        boardpath = opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr
        if not os.path.isdir(boardpath):
            os.makedirs(boardpath)
        TensorWriter = SummaryWriter(boardpath)

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

    # ==================================================build model==================================================
    model = get_model(args.modelname, args=args, opt=opt)
    opt.batch_size = args.batch_size * args.n_gpu

    tf_train = JointTransform3D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                                p_contr=0.5, p_gama=0.5, p_distor=0.0, color_jitter_params=None, long_mask=True)  # image reprocessing
    tf_val = JointTransform3D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    # tf_train, tf_val = None, None
    train_dataset = EchoVideoDataset(opt.data_path, opt.train_split, tf_train, img_size=args.encoder_input_size,frame_length=args.frame_length, point_numbers=args.point_numbers, disable_point_prompt=args.disable_point_prompt)
    val_dataset = EchoVideoDataset(opt.data_path, opt.val_split, tf_val, img_size=args.encoder_input_size,frame_length=args.frame_length, point_numbers=args.point_numbers, disable_point_prompt=args.disable_point_prompt)  # return image, mask, and filename
    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model.to(device)
    if opt.pre_trained:
        checkpoint = torch.load(opt.load_path)
        new_state_dict = {}
        for k,v in checkpoint.items():
            if k[:7] == 'module.':
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    if args.warmup:
        b_lr = args.base_lr / args.warmup_period
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        b_lr = args.base_lr
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    criterion = get_criterion(modelname=args.modelname, opt=opt)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    #  ========================================================================= begin to train the model ============================================================================
    iter_num = 0
    max_iterations = opt.epochs * len(trainloader)
    best_dice, loss_log, dice_log = 0.0, np.zeros(opt.epochs+1), np.zeros(opt.epochs+1)
    for epoch in range(opt.epochs):
        #  --------------------------------------------------------- training ---------------------------------------------------------
        model.train()
        train_losses = 0
        for batch_idx, (datapack) in enumerate(trainloader):
            imgs = datapack['image'].to(dtype = torch.float32, device=opt.device)
            masks = datapack['label'].to(dtype = torch.float32, device=opt.device)
            if args.disable_point_prompt:
                # pt[0]: b t point_num 2
                # pt[1]: t point_num
                pt = None
            else:
                pt = get_click_prompt(datapack, opt) 
            # video to image
            # b, t, c, h, w = imgs.shape
            # -------------------------------------------------------- forward --------------------------------------------------------
            pred = model(imgs, pt, None)
            # if masks.shape[1] == 10:
            #     masks = masks[:,[0,-1]]
            # semi supervised
            if opt.semi:
                train_loss = criterion(pred[:,[0,-1],0,:,:], masks[:,[0,-1]])
            # full supervised
            else:
                train_loss = criterion(pred[:,:,0], masks)
            # -------------------------------------------------------- backward -------------------------------------------------------
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_losses += train_loss.item()
            print(train_loss)
            # ------------------------------------------- adjust the learning rate when needed-----------------------------------------
            if args.warmup and iter_num < args.warmup_period:
                lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                    lr_ = args.base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
            iter_num = iter_num + 1

        #  -------------------------------------------------- log the train progress --------------------------------------------------
        print('epoch [{}/{}], train loss:{:.4f}'.format(epoch, opt.epochs, train_losses / (batch_idx + 1)))
        if args.keep_log:
            TensorWriter.add_scalar('train_loss', train_losses / (batch_idx + 1), epoch)
            TensorWriter.add_scalar('learning rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            loss_log[epoch] = train_losses / (batch_idx + 1)

        #  --------------------------------------------------------- evaluation ----------------------------------------------------------
        if epoch % opt.eval_freq == 0:
            model.eval()
            dices, mean_dice, _, val_losses = get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
            print('epoch [{}/{}], val loss:{:.4f}'.format(epoch, opt.epochs, val_losses))
            print('epoch [{}/{}], val dice:{:.4f}'.format(epoch, opt.epochs, mean_dice))
            if args.keep_log:
                TensorWriter.add_scalar('val_loss', val_losses, epoch)
                TensorWriter.add_scalar('dices', mean_dice, epoch)
                dice_log[epoch] = mean_dice
            if mean_dice > best_dice:
                best_dice = mean_dice
                timestr = time.strftime('%m%d%H%M')
                if not os.path.isdir(opt.save_path):
                    os.makedirs(opt.save_path)
                save_path = opt.save_path + args.modelname + opt.save_path_code + '%s' % timestr + '_' + str(epoch) + '_' + str(best_dice)
                torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
        if epoch % opt.save_freq == 0 or epoch == (opt.epochs-1):
            if not os.path.isdir(opt.save_path):
                os.makedirs(opt.save_path)
            save_path = opt.save_path + args.modelname + opt.save_path_code + '_' + str(epoch)
            torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
            # if args.keep_log:
            #     with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/trainloss.txt', 'w') as f:
            #         for i in range(len(loss_log)):
            #             f.write(str(loss_log[i])+'\n')
            #     with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/dice.txt', 'w') as f:
            #         for i in range(len(dice_log)):
            #             f.write(str(dice_log[i])+'\n')


if __name__ == '__main__':
    main()