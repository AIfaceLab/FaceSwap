import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import os
from tqdm import tqdm
import pytorch_ssim

from Model import Encoder, Decoder

from utils import *
from dataset import *
from LossFunction import MaskLoss, LossCnt
from torchvision import transforms


if __name__ == '__main__':
    num_epoch = 200000
    batch_size = 4
    src_device = "cuda:0"
    obj_device = "cuda:0"
    encoder_device = "cuda:0"
    maskloss_device = "cuda:0"
    mseloss_device = "cuda:0"
    input_size = 128
    len_iteration = 5
    # load datas
    src_image_paths = get_image_list(
        r"face_segmentation/results/data_src_girlB")
    obj_image_paths = get_image_list(
        r"face_segmentation/results/data_obj_girlA")

    # data
    data_transforms = transforms.Compose([
        # FlipHorizontally(),
        # RandomWarp(res=input_size)
        TransformDeepfakes(warp=True, transform=False,
                           is_border_replicate=True),
        RandomRotation(20)
    ])

    src_dataset = FaceData(src_image_paths, input_size, data_transforms)
    src_dataloader = DataLoader(
        src_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    obj_dataset = FaceData(obj_image_paths, input_size, data_transforms)
    obj_dataloader = DataLoader(
        obj_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    dataset = MergeDataSet(src_dataset, obj_dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)

    # initialize model， A represent src，B represent obj
    Encoder_AB = Encoder(shape=input_size).to(encoder_device)
    Decoder_A = Decoder(shape=input_size).to(src_device)
    Decoder_B = Decoder(shape=input_size).to(obj_device)

    # -------------------the different loss functions----------------------
    criterion_likely = pytorch_ssim.SSIM(window_size=11)
    criterion_pixel = nn.MSELoss()

    criterion_likely_mask = pytorch_ssim.SSIM(window_size=11)
    criterion_pixel_mask = nn.MSELoss().to(mseloss_device)
    criterion_cnt = LossCnt(device="cuda:0")
    criterion = MaskLoss(device=maskloss_device).to(maskloss_device)
    # ----------------------------------------------------------------------

    # Encoder_optimizer = torch.optim.Adam(
    #     Encoder_AB.parameters(), lr=5e-5, betas=(0.5, 0.999))
    # Decoder_A_optimizer = torch.optim.Adam(
    #     Decoder_A.parameters(), lr=5e-5, betas=(0.5, 0.999))
    # Decoder_B_optimizer = torch.optim.Adam(
    #     Decoder_B.parameters(), lr=5e-5, betas=(0.5, 0.999))

    AE_optimizer = torch.optim.Adam(params=list(Encoder_AB.parameters())+list(Decoder_A.parameters())+list(Decoder_B.parameters()),
                                    lr=5e-5, betas=(0.5, 0.999))

    Logger = LogModel(input_size, model_name="girlB2A")
    iter_epoch = Logger.load_model(
        Encoder_AB, Decoder_A, Decoder_B)
    # iter_epoch = 0

    Encoder_AB.train()
    Decoder_A.train()
    Decoder_B.train()

    loss_visualize = LossVisualize(
        title="Loss of Model", resolution=input_size, loss_name="girlB2A")

    for epoch in tqdm(range(iter_epoch, num_epoch)):
        for i, (img_src, img_obj) in enumerate(dataloader):
            src_out, src_mask = Decoder_A(
                Encoder_AB(img_src['rgb'].cuda()))  # 3*128*128

            obj_out, obj_mask = Decoder_B(Encoder_AB(img_obj['rgb'].cuda()))

            # +criterion(img_src['mask'].cpu(), src_out.cpu(), img_src['rgb'].cpu())
            # criterion_cnt((src_out.cpu()).cuda()*255, (img_src['rgb']).cuda()*255).cpu()
            loss_mask_src = criterion_pixel_mask(
                src_mask.to(mseloss_device), img_src['mask_label'].to(mseloss_device))
            loss_src = criterion(img_src['mask_label'].to(maskloss_device),
                                 src_out.to(maskloss_device), img_src['rgb_label'].to(maskloss_device)) + loss_mask_src
            # +criterion(img_obj['mask'].cpu(), obj_out.cpu(), img_obj['rgb'].cpu())
            # criterion_cnt((obj_out.cpu()).cuda()*255, (img_obj['rgb']).cuda()*255).cpu()
            loss_mask_obj = criterion_pixel_mask(
                obj_mask.to(mseloss_device), img_obj['mask_label'].to(mseloss_device))
            loss_obj = criterion(img_obj['mask_label'].to(maskloss_device),
                                 obj_out, img_obj['rgb_label'].to(maskloss_device))+loss_mask_obj
            loss_mask = loss_mask_obj + loss_mask_src
            # loss = loss_src+loss_obj + loss_mask

            # log loss curve
            loss_visualize.log(
                {"loss_src": loss_src.item(), "loss_obj": loss_obj.item()})

            AE_optimizer.zero_grad()
            loss_src.backward()
            AE_optimizer.step()

            AE_optimizer.zero_grad()
            loss_obj.backward()
            AE_optimizer.step()
            if i % 20 == 0:
                print('Loss of src is {},Loss of obj is {}'.format(
                    loss_src.item(), loss_obj.item()))

                # src
                visualize_output("img_src_input", img_src['rgb'].cpu())
                visualize_output("img_src_label", img_src['rgb_label'].cpu())
                visualize_output("src_out", src_out.cpu())
                visualize_output("label_src_mask", img_src['mask_label'].cpu())
                visualize_output("out_src_mask", src_mask.cpu())
                # obj
                visualize_output("img_obj_input", img_obj['rgb'].cpu())
                visualize_output("img_obj_label", img_obj['rgb_label'].cpu())
                visualize_output("obj_out", obj_out.cpu())
                visualize_output("label_obj_mask", img_obj['mask_label'].cpu())
                visualize_output("out_obj_mask", obj_mask.cpu())
                # swapped
                swap_src, _ = Decoder_B(Encoder_AB(img_src['rgb'].cuda()))
                swap_obj, _ = Decoder_A(Encoder_AB(img_obj['rgb'].cuda()))
                visualize_output("swapsrc_out", swap_src.cpu())
                visualize_output("swapobj_out", swap_obj.cpu())

        Logger.log_model(epoch+iter_epoch, Encoder_AB.state_dict(), Decoder_A.state_dict(),
                         Decoder_B.state_dict())
        loss_visualize.save_loss2file()
