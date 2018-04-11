import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from .fpn import fpn50
from .layers.groupnorm import GroupNorm2d
from .layers.proposals import Proposals
from .layers.mask_head_proposals import MaskHeadProposals
from .layers.roi import Roi
from .layers.loss import FocalLoss, MaskLoss
from .utils import crop_masks, place_masks

class RetinaNet(nn.Module):
    def __init__(self, fpn_factory=fpn50, num_classes=1, num_anchors=9, max_instances = 320, pooling_size=14):
        super(RetinaNet, self).__init__()
        self.pooling_size = pooling_size
        self.predicting = False
        self.max_instances = max_instances
        self.fpn = fpn_factory()
        self.proposals = Proposals(max_instances=max_instances)
        self.mask_head_proposals = MaskHeadProposals(max_instances=max_instances)
        self.roi = Roi(max_instances=max_instances,pooling_size=pooling_size)
        self.num_classes = num_classes + 1 # including background
        self.num_anchors = num_anchors
        self.cls_head = self._build_head(num_anchors * self.num_classes)
        self.box_head = self._build_head(num_anchors * 4)
        self.mask_head = self._build_mask_head()
        self.detection_loss = FocalLoss(self.num_classes)
        self.mask_loss = MaskLoss()

        self._init_cls_head(self.cls_head)
        self._init_head(self.box_head)


    def predict(self):
        self.eval()
        self.predicting = True


    def train(self, mode=True):
        self.predicting = False
        super(RetinaNet, self).train(mode)


    def freeze_mask(self):
        for param in self.mask_head.parameters():
            param.requires_grad = False


    def unfreeze_mask(self):
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

        if not self.predicting:
            cls_loss, box_loss = self.detection_loss(cls_preds, classes, box_preds, boxes, input_size)

        cls_proposals, box_proposals, scores = self.proposals(cls_preds, box_preds, input_size)

        if self.training:
            cls_proposals, box_proposals, scores = self.mask_head_proposals(cls_proposals, classes, box_proposals, boxes, scores)

        roi_feature_maps = self.roi(feature_maps[0], box_proposals, input_size)
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

        masks = self._prepare_gt_masks(masks, box_proposals)
        mask_loss = self.mask_loss(mask_preds, masks, scores)
        mask_preds = place_masks(mask_preds, box_proposals, input_size)

        total_loss = cls_loss + box_loss + mask_loss

        return cls_proposals, box_proposals, mask_preds, cls_loss, box_loss, mask_loss, total_loss


    def _prepare_gt_masks(self, masks, boxes):
        num_masks = masks.shape[1]
        masks = crop_masks(masks, boxes[:,:num_masks], pooling_size=self.pooling_size)
        pad_size = self.max_instances - masks.shape[1]
        return F.pad(masks, (0,0,0,0,0,pad_size))


    def _build_head(self, num_planes):
        layers = []

        for _ in range(4):
            layers.append(nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1))
            layers.append(GroupNorm2d(256))
            layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(256, num_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)


    def _build_mask_head(self):
        layers = []

        for _ in range(4):
            layers.append(nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1))
            layers.append(GroupNorm2d(256))
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

