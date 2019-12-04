#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

# im is rgb


def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(
        vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)

    vis_parsing_anno_color = np.zeros(
        (vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    mask = np.zeros(
        (vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]), dtype=np.uint8)
    num_of_class = np.max(vis_parsing_anno)

    idx = 11
    for pi in range(1, num_of_class + 1):
        index = np.where((vis_parsing_anno <= 5) & (
            vis_parsing_anno >= 1) | ((vis_parsing_anno >= 10) & (vis_parsing_anno <= 13)))
        mask[index[0], index[1]] = 1
    result = merge(im, mask)
    cv2.imwrite(save_path[:-4] + '.png', result)


def evaluate(respth='./results/data_src', dspth='../data', cp='79999_iter.pth'):
    respth = osp.join(os.path.abspath(os.path.dirname(__file__)), respth)
    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    model_path = osp.join(os.path.abspath(os.path.dirname(__file__)), cp)
    data_path = osp.join(os.path.abspath(os.path.dirname(__file__)), dspth)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(data_path):
            img = Image.open(osp.join(data_path, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            print(np.unique(parsing))

            vis_parsing_maps(image, parsing, stride=1, save_im=True,
                             save_path=osp.join(respth, image_path))
            # break


def merge(img_1, mask):
    r_channel, g_channel, b_channel = cv2.split(img_1)
    if mask is not None:
        alpha_channel = np.ones(mask.shape, dtype=img_1.dtype)
        alpha_channel *= mask*255
    else:
        alpha_channel = np.zeros(img_1.shape[:2], dtype=img_1.dtype)
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_BGRA


if __name__ == "__main__":
    evaluate(dspth='../data/test',
             respth='./results/test', cp='79999_iter.pth')
