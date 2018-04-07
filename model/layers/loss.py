import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.utils import one_hot_embedding


class FocalLoss(nn.Module):
    def __init__(self, num_classes=1):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.include_mask = True


    def focal_loss(self, x,y):
        alpha = 0.25
        gamma = 2

        t = one_hot_embedding(y.data.cpu(), 1+self.num_classes)  # [N,2]
        t = t[:,1:]  # exclude background
        t = Variable(t).cuda()  # [N,2]

        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        res =  F.binary_cross_entropy_with_logits(x, t, w, size_average=False)
        return res


    def forward(self, cls_preds, cls_targets, cls_proposals, box_preds, box_targets, mask_preds, mask_targets):
        """ Compute focal loss for boxes and targets
        """
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0
        num_pos = pos.data.long().sum()

        mask = pos.unsqueeze(2).expand_as(box_preds)
        masked_box_preds = box_preds[mask].view(-1,4)
        masked_box_targets = box_targets[mask].view(-1,4)
        box_loss = F.smooth_l1_loss(masked_box_preds, masked_box_targets, size_average=False)

        pos_neg = cls_targets > -1
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1,self.num_classes)
        cls_loss = self.focal_loss(masked_cls_preds, cls_targets[pos_neg])

        mask_loss = Variable(box_loss.data.new(1).fill_(0))
        if self.include_mask:
            mask_size = mask_preds.shape[2]
            max_detections = int(cls_proposals.sum(dim=1).max())
            pad_size = max_detections - mask_targets.shape[1]
            mask_targets = F.pad(mask_targets, (0,0,0,0,0,pad_size))
            pad_size = max_detections - mask_preds.shape[1]
            mask_preds = F.pad(mask_preds, (0,0,0,0,0,pad_size))
            mask_loss = F.binary_cross_entropy_with_logits(mask_preds, mask_targets, size_average=False)

        avg_loss = (box_loss + cls_loss + mask_loss)/num_pos
        return cls_loss, box_loss, mask_loss, avg_loss

