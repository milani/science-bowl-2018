import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.utils import one_hot_embedding

class FocalLoss(nn.Module):
    def __init__(self, num_classes=1):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes


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


    def forward(self, cls_preds, cls_targets, box_preds, box_targets):
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

        loss = (box_loss+cls_loss)/num_pos
        return loss

