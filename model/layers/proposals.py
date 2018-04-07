import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.utils import box_nms
from model.layers.anchors import Anchors


class Proposals(Anchors):
    def __init__(self, max_instances = 320):
        super(Proposals,self).__init__()
        self.max_instances = max_instances


    def forward(self, cls_preds:Variable, box_preds:Variable, input_size:torch.Size):
        CLS_THRESH = 0.3
        NMS_THRESH = 0.6
        max_instances = self.max_instances
        box_preds = box_preds.data
        cls_preds = cls_preds.data
        dtype = cls_preds.type()
        batch_size = cls_preds.shape[0]
        scores, classes, boxes = self.deanchorize(cls_preds, box_preds, input_size)
        box_results = box_preds.new(batch_size, max_instances, 4).fill_(0)
        class_results = box_preds.new(batch_size, max_instances).fill_(0)
        score_results = box_preds.new(batch_size, max_instances).fill_(0)

        for b in range(batch_size):
            ids = scores[b,:] > CLS_THRESH
            ids = ids.nonzero().squeeze()
            if len(ids) == 0:
                continue
            keep = box_nms(boxes[b][ids], scores[b][ids], threshold=NMS_THRESH).type(dtype).long()
            keep_ids = ids[keep]
            if len(keep_ids) > max_instances:
                keep_ids = keep_ids[:max_instances]
            box_results[b][:len(keep_ids),:] = boxes[b][keep_ids]
            class_results[b][:len(keep_ids)] = classes[b][keep_ids] + 1
            score_results[b][:len(keep_ids)] = scores[b][keep_ids]

        return Variable(class_results, requires_grad=False), Variable(box_results, requires_grad=False), Variable(score_results, requires_grad=False)


    def deanchorize(self, cls_preds, box_preds, input_size):
        dtype = box_preds.type()
        batch_size = box_preds.shape[0]
        if isinstance(input_size,torch.Size):
            input_size = tuple(input_size)
        input_size = torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size).type(dtype)
        anchor_boxes = anchor_boxes.unsqueeze(0).expand(batch_size,*anchor_boxes.shape)

        box_xy = box_preds[:,:,:2]
        box_wh = box_preds[:,:,2:]
        xy = box_xy * anchor_boxes[:,:,2:] + anchor_boxes[:,:,:2]
        wh = box_wh.exp() * anchor_boxes[:,:,2:]
        boxes = torch.cat([xy-wh/2, xy+wh/2], 2)
        scores, classes = cls_preds.sigmoid().max(2)

        return scores, classes, boxes