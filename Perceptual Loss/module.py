# perceptual loss
import torch
import torch.nn as nn
import torchvision.models as models


class PerceptualLoss(nn.Module):
    def __init__(self, loss_func, layer_indexs, device):
        super(PerceptualLoss, self).__init__()
        self.device = device
        self.loss_func = loss_func
        self.layer_indexs=layer_indexs

    def get_feature_module(self, layer_index, device=None):
        vgg = models.vgg16(pretrained=True, progress=True).features
        vgg.eval()
        # freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False
        feature_module = vgg[0 : layer_index + 1]
        feature_module.to(device)
        return feature_module

    def vgg_loss(self, feature_module, loss_func, target, inputs):
        out_trg = feature_module(target)
        out_ipt = feature_module(inputs)
        loss = loss_func(out_trg, out_ipt)
        return loss
        
    def forward(self, target, inputs):
        loss = 0
        for index in self.layer_indexs:
            feature_module = self.get_feature_module(index, self.device)
            loss += self.vgg_loss(feature_module, self.loss_func, target, inputs)
        return loss
