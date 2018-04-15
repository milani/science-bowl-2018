import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.utils import one_hot_embedding
from model.layers.anchors import Anchors


class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()


    def forward(self, mask_preds, masks, scores):
        num_masks, mask_height, mask_width = masks.shape[1:]
        scores = scores.view(1, -1).squeeze(0)
        masks = masks.view(1, -1, mask_height, mask_width).squeeze(0)
        positive_idx = scores > 0
        positive_idx = positive_idx.unsqueeze(1).unsqueeze(1) # broadcastable
        masks = torch.masked_select(masks, positive_idx)
        masks[masks >= 0.5] = 1
        masks[masks < 0.5] = 0

        pad_size = num_masks - mask_preds.shape[1]
        mask_preds = F.pad(mask_preds, (0,0,0,0,0,pad_size))
        mask_preds = mask_preds.view(1, -1, mask_height, mask_width)
        mask_preds = torch.masked_select(mask_preds, positive_idx)
        return F.binary_cross_entropy_with_logits(mask_preds, masks, size_average=False) / positive_idx.float().sum()


class FocalLoss(nn.Module):
    def __init__(self, num_classes=2):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.anchorize = Anchors()


    def focal_loss(self, x,y):
        alpha = 0.25
        gamma = 2

        t = one_hot_embedding(y.data.cpu(), self.num_classes)  # [N,2]

        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        res =  F.binary_cross_entropy_with_logits(x, t, w, size_average=False)
        return res


    def forward(self, cls_preds, cls_targets, box_preds, box_targets, input_size):
        cls_targets, box_targets = self.anchorize(cls_targets, box_targets, input_size)
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0
        num_pos = pos.data.long().sum()

        # box loss
        mask = pos.unsqueeze(2).expand_as(box_preds)
        masked_box_preds = box_preds[mask].view(-1,4)
        masked_box_targets = box_targets[mask].view(-1,4)
        box_loss = F.smooth_l1_loss(masked_box_preds, masked_box_targets, size_average=False)

        # class loss
        pos_neg = cls_targets > -1
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1,self.num_classes)
        cls_loss = self.focal_loss(masked_cls_preds, cls_targets[pos_neg])
        return cls_loss, box_loss

