import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from model.fpn import fpn50
from model.layers.proposals import Proposals
from model.layers.roi import Roi
from model.layers.loss import FocalLoss
from model.layers.anchors import Anchors
from model.utils import crop_masks, place_masks

class RetinaNet(nn.Module):
    def __init__(self, fpn_factory=fpn50, num_classes=1, num_anchors=9, max_instances = 320, pooling_size=21):
        super(RetinaNet, self).__init__()
        self.pooling_size = pooling_size
        self.predicting = False
        self.max_instances = max_instances
        self.fpn = fpn_factory()
        self.anchorize = Anchors()
        self.proposals = Proposals(max_instances=max_instances)
        self.roi = Roi(max_instances=max_instances,pooling_size=pooling_size)
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.cls_head = self._build_head(num_anchors * num_classes)
        self.box_head = self._build_head(num_anchors * 4)
        self.mask_head = self._build_mask_head()
        self.loss = FocalLoss(num_classes)

        self._init_cls_head(self.cls_head)
        self._init_head(self.box_head)


    def predict(self):
        self.eval()
        self.predicting = True


    def train(self, mode=True):
        self.predicting = False
        super(RetinaNet, self).train(mode)


    def freeze_mask(self):
        self.loss.include_mask = False
        for param in self.mask_head.parameters():
            param.requires_grad = False


    def unfreeze_mask(self):
        self.loss.include_mask = True
        for param in self.mask_head.parameters():
            param.requires_grad = True


    def forward(self, imgs, classes=None, boxes=None, masks=None):
        input_size = imgs.shape[2:]
        feature_maps = self.fpn(imgs)
        box_preds = []
        cls_preds = []

        for fm in feature_maps:
            cls_pred = self.cls_head(fm)
            box_pred = self.box_head(fm)
            # [N,9*2,H,W] -> [N,H,W,9*2] -> [N,H*W*9,2]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(imgs.size(0), -1, self.num_classes)
            # [N,9*4,H,W] -> [N,H,W,9*4] -> [N,H*W*9,4]
            box_pred = box_pred.permute(0, 2, 3, 1).contiguous().view(imgs.size(0), -1, 4)

            cls_preds.append(cls_pred)
            box_preds.append(box_pred)

        cls_preds = torch.cat(cls_preds,1)
        box_preds = torch.cat(box_preds,1)

        cls_proposals, box_proposals, scores = self.proposals(cls_preds, box_preds, input_size)

        # TODO use a better approach for mixing ground truth boxes
        if not self.predicting:
            last_index = min(int(cls_proposals.sum(dim=1).max()), self.max_instances - 1)
            for i in range(masks.shape[0]):
                scores[i,last_index] = 1
                cls_proposals[i,last_index] = 1
                box_proposals[i,last_index,:] = boxes.data[i,0,:]

        roi_feature_maps = self.roi(feature_maps[0], box_proposals, scores, input_size)
        mask_preds = []
        for fm in roi_feature_maps:
            mask_pred = self.mask_head(fm)
            mask_pred = mask_pred.permute(1,0,2,3) # MASKS x 1 x 7 x 7 -> 1 x MASKS x 7 x 7
            pad = self.max_instances - mask_pred.shape[1]
            mask_preds.append(F.pad(mask_pred,(0,0,0,0,0,pad)))
        mask_preds = torch.cat(mask_preds,0)

        if self.predicting:
            mask_preds = place_masks(mask_preds, box_proposals, input_size)
            return cls_proposals, box_proposals, mask_preds

        masks = crop_masks(masks, boxes, pooling_size=self.pooling_size)
        classes, boxes = self.anchorize(classes, boxes, input_size)

        losses = self.loss(cls_preds, classes, cls_proposals, box_preds, boxes, mask_preds, masks)
        cls_loss, box_loss, mask_loss, total_loss = losses

        mask_preds = place_masks(mask_preds, box_proposals, input_size)

        return cls_proposals, box_proposals, mask_preds, cls_loss, box_loss, mask_loss, total_loss


    def _build_head(self, num_planes):
        layers = []

        for _ in range(4):
            layers.append(nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(256, num_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)


    def _build_mask_head(self):
        layers = []

        for _ in range(3):
            layers.append(nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0))
        return nn.Sequential(*layers)


    def _init_cls_head(self, head):
        pi = 0.01
        self._init_head(head)
        last_layer = list(head.modules())[-1]
        value = np.log(pi) - np.log(1-pi)
        last_layer.bias.data.fill_(value)


    def _init_head(self, head):
        for m in head.modules():
            if isinstance(m, nn.Conv2d):
                m.bias.data.zero_()
                m.weight.data.normal_(0,0.01)

