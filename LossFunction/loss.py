import torch
import torch.nn as nn
import numpy as np
import imp
from torchvision.models import vgg19
from .CroppedVgg import Cropped_VGG19
import torchvision
import os
from .Gaussian_Blur import GaussianSmoothing
from utils import visualize_output
from torch.nn import functional as F


class MaskLoss(nn.Module):
    def __init__(self, is_mse=True):
        super(MaskLoss, self).__init__()
        self.is_mes = is_mse

    def forward(self, mask, image, label):
        if self.is_mes:
            diff2 = 50*(torch.flatten(label) - torch.flatten(image)) ** 2.0
            # sum2 = 0.0
            # num = 0
            # mask = torch.unsqueeze(mask, 1)
            # -----------------------gaussian filter mask-------------------------------------------
            gaussian_blur = GaussianSmoothing(1, 5, 2.0)
            mask = F.pad(mask, (2, 2, 2, 2), mode='reflect')
            mask = gaussian_blur(mask)
            temp_mask = mask.repeat(1, 3, 1, 1)
            visualize_output("temp_mask", temp_mask)
            # --------------------------------------------------------------------------------------
            flat_mask = torch.flatten(temp_mask)
            loss = torch.mean(diff2*flat_mask)
            # np.mean(50*np.square(label*mask, image*mask))
            # loss = torch.mean(10*(label-image)**2)
        else:
            pass
        return loss


class LossCnt(nn.Module):
    def __init__(self, device='cpu', VGGFace_body_path='Pytorch_VGGFACE_IR.py', VGGFace_weight_path='Pytorch_VGGFACE.pth'):
        super(LossCnt, self).__init__()

        self.VGG19 = vgg19(pretrained=True)
        self.VGG19.eval()
        self.VGG19.to(device)

        VGGFace_body_path = os.path.join(
            os.path.dirname(__file__), VGGFace_body_path)
        VGGFace_weight_path = os.path.join(
            os.path.dirname(__file__), VGGFace_weight_path)
        # MainModel = imp.load_source('MainModel', VGGFace_body_path)
        full_VGGFace = torch.load(VGGFace_weight_path, map_location='cpu')
        cropped_VGGFace = Cropped_VGG19()
        #        cropped_VGGFace.load_state_dict(full_VGGFace.state_dict(), strict = False)
        cropped_VGGFace.load_state_dict(full_VGGFace, strict=False)
        self.VGGFace = cropped_VGGFace
        self.VGGFace.eval()
        self.VGGFace.to(device)

    def forward(self, x, x_hat, vgg19_weight=1e-2, vggface_weight=2e-3):
        l1_loss = nn.L1Loss()

        """Retrieve vggface feature maps"""
        with torch.no_grad():  # no need for gradient compute
            # returns a list of feature maps at desired layers
            vgg_x_features = self.VGGFace(x)

        vgg_xhat_features = self.VGGFace(x_hat)

        lossface = 0
        for x_feat, xhat_feat in zip(vgg_x_features, vgg_xhat_features):
            lossface += l1_loss(x_feat, xhat_feat)

        """Retrieve vggface feature maps"""

        # define hook
        def vgg_x_hook(module, input, output):
            output.detach_()  # no gradient compute
            vgg_x_features.append(output)

        def vgg_xhat_hook(module, input, output):
            vgg_xhat_features.append(output)

        vgg_x_features = []
        vgg_xhat_features = []

        vgg_x_handles = []

        # idxes of conv layers in VGG19 cf.paper
        conv_idx_list = [2, 7, 12, 21, 30]
        conv_idx_iter = 0

        # place hooks
        for i, m in enumerate(self.VGG19.features.modules()):
            if i == conv_idx_list[conv_idx_iter]:
                if conv_idx_iter < len(conv_idx_list) - 1:
                    conv_idx_iter += 1
                vgg_x_handles.append(m.register_forward_hook(vgg_x_hook))

        # run model for x
        self.VGG19(x)

        # retrieve features for x
        for h in vgg_x_handles:
            h.remove()

        # retrieve features for x_hat
        conv_idx_iter = 0
        for i, m in enumerate(self.VGG19.modules()):
            if i <= 30:  # 30 is last conv layer
                if type(m) is not torch.nn.Sequential and type(m) is not torchvision.models.vgg.VGG:
                    # only pass through nn.module layers
                    if i == conv_idx_list[conv_idx_iter]:
                        if conv_idx_iter < len(conv_idx_list) - 1:
                            conv_idx_iter += 1
                        x_hat = m(x_hat)
                        vgg_xhat_features.append(x_hat)
                        x_hat.detach_()  # reset gradient from output of conv layer
                    else:
                        x_hat = m(x_hat)

        loss19 = 0
        for x_feat, xhat_feat in zip(vgg_x_features, vgg_xhat_features):
            loss19 += l1_loss(x_feat, xhat_feat)

        loss = vgg19_weight * loss19 + vggface_weight * lossface

        return loss
