import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, predicted, target):

        c_losses = []
        for c in range(predicted.shape[1]):
            cross_entropy = F.binary_cross_entropy(predicted[:,c,:,:], target[:,c,:,:])
            focal_loss = cross_entropy * (1 - torch.exp(-cross_entropy)) ** self.gamma
            c_losses.append(focal_loss)

        c_losses = torch.stack(c_losses, dim=0)
        weighted_losses = 1 * c_losses
        loss = weighted_losses.sum()
        return loss


class WeightedBCE(torch.nn.Module):
    def __init__(self, weights_per_class):
        super(WeightedBCE, self).__init__()
        self.weights_per_class = weights_per_class

    def forward(self, predicted, target):

        # put different weights on different classes
        weights = torch.tensor(self.weights_per_class).to(predicted.device)
        c_losses = []
        for c in range(predicted.shape[1]):
            c_losses.append(F.binary_cross_entropy(predicted[:,c,:,:], target[:,c,:,:]))

        c_losses = torch.stack(c_losses, dim=0)
        weighted_losses = weights * c_losses
        loss = weighted_losses.sum()
        return loss


class ClassBalanceLoss(torch.nn.Module):
    def __init__(self, pixel_per_classes_distr, beta=0.9999, gamma=2.0):
        super(ClassBalanceLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.pixel_per_classes_distr = pixel_per_classes_distr
        self.loss_type="cross_entropy"

    def forward(self, predicted, target):
        effective_num = 1.0 - np.power(self.beta, self.pixel_per_classes_distr)
        weights = (1.0 - self.beta) / effective_num
        weights = torch.tensor(weights / np.sum(weights)).float().to(predicted.device)
        c_losses = []
        for c in range(predicted.shape[1]):
            
            if self.loss_type == "cross_entropy":
                lossona = F.binary_cross_entropy(predicted[:,c,:,:], target[:,c,:,:])
            
            elif self.loss_type == "focal":
                cross_entropy = F.binary_cross_entropy(predicted[:,c,:,:], target[:,c,:,:])
                lossona = cross_entropy * (1 - torch.exp(-cross_entropy)) ** self.gamma
            
            c_losses.append(lossona)

        c_losses = torch.stack(c_losses, dim=0)
        weighted_losses = weights * c_losses
        loss = weighted_losses.sum()
        return loss


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predicted, target):
        # Flatten the predictions and targets
        predicted_flat = predicted.flatten()
        target_flat = target.flatten()

        # Intersection and Union
        intersection = torch.sum(predicted_flat * target_flat)
        union = torch.sum(predicted_flat) + torch.sum(target_flat)

        # Dice Coefficient
        dice_coefficient = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice Loss
        dice_loss = 1.0 - dice_coefficient

        return dice_loss
