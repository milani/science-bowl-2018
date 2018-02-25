import torch
import torch.nn as nn
import numpy as np
from model.fpn import fpn50

class RetinaNet(nn.Module):
    def __init__(self, fpn_factory=fpn50, num_classes=1, num_anchors=9):
        super(RetinaNet, self).__init__()
        self.fpn = fpn_factory()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.cls_head = self._build_head(num_anchors * num_classes)
        self.box_head = self._build_head(num_anchors * 4)

        self._init_cls_head(self.cls_head)
        self._init_head(self.box_head)


    def forward(self, x):
        feature_maps = self.fpn(x)
        box_preds = []
        cls_preds = []

        for fm in feature_maps:
            cls_pred = self.cls_head(fm)
            box_pred = self.box_head(fm)
            # [N,9*2,H,W] -> [N,H,W,9*2] -> [N,H*W*9,2]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
            # [N,9*4,H,W] -> [N,H,W,9*4] -> [N,H*W*9,4]
            box_pred = box_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)

            cls_preds.append(cls_pred)
            box_preds.append(box_pred)

        return torch.cat(cls_preds,1), torch.cat(box_preds,1)


    def _build_head(self, num_planes):
        layers = []

        for _ in range(4):
            layers.append(nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(256, num_planes, kernel_size=3, stride=1, padding=1))
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

