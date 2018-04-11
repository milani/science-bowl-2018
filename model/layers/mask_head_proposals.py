import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..utils import box_nms

class MaskHeadProposals(nn.Module):
    def __init__(self, nms_thr=0.3, max_instances=320):
        super(MaskHeadProposals, self).__init__()
        self.nms_thr = nms_thr
        self.max_instances = max_instances

    def forward(self, cls_proposals, gt_classes, box_proposals, gt_boxes, proposal_scores):
        batch_size, num_boxes = gt_boxes.shape[:2]
        gt_scores = gt_classes

        cls_proposals = torch.cat([gt_classes.data, cls_proposals.data], dim=1)
        box_proposals = torch.cat([gt_boxes.data, box_proposals.data], dim=1)
        proposal_scores = torch.cat([gt_scores.data, proposal_scores.data], dim=1)

        new_classes = []
        new_boxes = []
        new_scores = []

        for b in range(batch_size):
            keep = box_nms(box_proposals[b], proposal_scores[b], threshold=self.nms_thr)
            keep, _ = keep.sort()
            keep = keep[:self.max_instances]

            pad_size = self.max_instances - len(keep)
            tmp_classes = self._pad(cls_proposals[b][keep], (0, pad_size))
            tmp_boxes = self._pad(box_proposals[b][keep], (0, 0, 0, pad_size))
            tmp_scores = self._pad(proposal_scores[b][keep], (0, pad_size))

            new_classes.append(tmp_classes.unsqueeze(0))
            new_boxes.append(tmp_boxes.unsqueeze(0))
            new_scores.append(tmp_scores.unsqueeze(0))

        new_classes = torch.cat(new_classes, dim=0)
        new_boxes = torch.cat(new_boxes, dim=0)
        new_scores = torch.cat(new_scores, dim=0)
        return new_classes, new_boxes, new_scores


    def _pad(self, tensor, pad_pattern):
        tensor = Variable(tensor, requires_grad=False)
        return F.pad(tensor, pad_pattern)
