import torch
import pdb
import math
from model.utils import meshgrid, box_iou, change_box_order, box_nms

class Anchorizer(object):
    def __init__(self):
        super(Anchorizer,self).__init__()
        self.anchor_areas = [8*8, 32*32, 128*128, 512*512]
        self.aspect_ratios = [1/2., 1/1., 2/1.]
        self.scale_ratios = [1., 2., pow(2, 2/3.)]
        self.anchor_wh = self._get_anchor_wh()


    def _get_anchor_wh(self):
        anchor_wh = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:
                h = math.sqrt(s/ar)
                w = ar * h
                for sr in self.scale_ratios:
                    anchor_h = h*sr
                    anchor_w = w*sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        return torch.Tensor(anchor_wh).view(num_fms, -1, 2)


    def _get_anchor_boxes(self, input_size):
        num_fms = len(self.anchor_areas)
        fm_sizes = [(input_size/pow(2.,i+2)).ceil() for i in range(num_fms)]

        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = input_size / fm_size
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            xy = meshgrid(fm_w, fm_h) + 0.5
            xy = (xy*grid_size).view(fm_h, fm_w, 1, 2).expand(fm_h, fm_w, 9, 2)
            wh = self.anchor_wh[i].view(1,1,9,2).expand(fm_h,fm_w,9,2)
            box = torch.cat([xy,wh], 3)  # [x,y,w,h]
            boxes.append(box.view(-1,4))
        return torch.cat(boxes, 0)


    def encode(self, labels, boxes, input_size):
        dtype = boxes.type()
        batch_size = boxes.shape[0]
        if isinstance(input_size,torch.Size):
            input_size = tuple(input_size)
        input_size = torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size).type(dtype)
        anchor_boxes = anchor_boxes.unsqueeze(0).expand(batch_size,*anchor_boxes.shape)

        boxes = change_box_order(boxes, 'xyxy2xywh')
        ious = box_iou(anchor_boxes, boxes, order='xywh')
        max_ious, max_ids = ious.max(2)
        boxes = torch.stack([boxes[b][max_ids[b,:]] for b in range(boxes.shape[0])])

        loc_xy = (boxes[:,:,:2]-anchor_boxes[:,:,:2]) / anchor_boxes[:,:,2:]
        loc_wh = torch.log(boxes[:,:,2:]/anchor_boxes[:,:,2:])
        loc_targets = torch.cat([loc_xy,loc_wh], 2)
        cls_targets = torch.stack([labels[b][max_ids[b,:]] for b in range(labels.shape[0])])
        cls_targets[max_ious<0.5] = 0
        ignore = (max_ious>0.4) & (max_ious<0.5)
        cls_targets[ignore] = -1
        return cls_targets, loc_targets


    def decode(self, cls_preds, box_preds, input_size):
        CLS_THRESH = 0.1
        NMS_THRESH = 0.1
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
        score, classes = cls_preds.sigmoid().max(2)
        box_results = []
        class_results = []
        for b in range(batch_size):
            ids = score[b,:] > CLS_THRESH
            ids = ids.nonzero().squeeze()
            if len(ids) == 0:
                class_results.append([])
                box_results.append([])
                continue
            keep = box_nms(boxes[b][ids], score[b][ids], threshold=0.1).type(dtype).long()
            box_results.append(boxes[b][ids[keep]])
            class_results.append(classes[b][ids[keep]])

        return class_results, box_results
