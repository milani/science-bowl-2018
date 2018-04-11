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
        self.softmax = nn.Softmax(dim=2)


    def forward(self, cls_preds:Variable, box_preds:Variable, input_size:torch.Size):
        CLS_THRESH = 0.6
        NMS_THRESH = 0.8
        pre_nms = 6000
        max_instances = self.max_instances
        cls_probs = self.softmax(cls_preds).data
        box_preds = box_preds.data
        cls_preds = cls_preds.data
        batch_size = cls_preds.shape[0]
        scores, classes, boxes = self.deanchorize(cls_probs, box_preds, input_size)

        box_results = box_preds.new(batch_size, max_instances, 4).fill_(0)
        class_results = box_preds.new(batch_size, max_instances).fill_(0)
        score_results = box_preds.new(batch_size, max_instances).fill_(0)

        for b in range(batch_size):
            sorted_scores, sorted_idx  = scores[b,:].sort()
            ids = sorted_idx[sorted_scores > CLS_THRESH]
            ids = ids[:pre_nms]
            keep = box_nms(boxes[b][ids], scores[b][ids], threshold=NMS_THRESH)
            keep, _ = keep.sort()
            keep_ids = ids[keep[:max_instances]]

            box_results[b][:len(keep_ids),:] = boxes[b][keep_ids]
            class_results[b][:len(keep_ids)] = classes[b][keep_ids]
            score_results[b][:len(keep_ids)] = scores[b][keep_ids]

        return Variable(class_results, requires_grad=False), Variable(box_results, requires_grad=False), Variable(score_results, requires_grad=False)


    def deanchorize(self, cls_probs, box_preds, input_size):
        dtype = box_preds.type()
        batch_size = box_preds.shape[0]
        if isinstance(input_size,torch.Size):
            input_size = tuple(input_size)
        h, w = input_size
        input_size = torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size).type(dtype)
        anchor_boxes = anchor_boxes.unsqueeze(0).expand(batch_size,*anchor_boxes.shape)

        scores, classes = cls_probs.max(2)

        box_xy = box_preds[:,:,:2]
        box_wh = box_preds[:,:,2:]
        xy = box_xy * anchor_boxes[:,:,2:] + anchor_boxes[:,:,:2]
        wh = box_wh.exp() * anchor_boxes[:,:,2:]
        boxes = torch.cat([xy-wh/2, xy+wh/2], 2)
        boxes[:,:,0:2].clamp_(min=0)
        boxes[:,:,2].clamp_(max=w)
        boxes[:,:,3].clamp_(max=h)

        return scores, classes, boxes
