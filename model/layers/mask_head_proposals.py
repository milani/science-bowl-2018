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
        dtype = cls_proposals.data.type()
        batch_size, num_boxes = gt_boxes.shape[:2]
        gt_scores = gt_classes

        # let's put background proposals first for nms
        pos_mask = (cls_proposals > 0)
        proposal_scores[pos_mask] = 0

        cls_proposals = torch.cat([gt_classes.data, cls_proposals.data], dim=1)
        box_proposals = torch.cat([gt_boxes.data, box_proposals.data], dim=1)
        proposal_scores = torch.cat([gt_scores.data, proposal_scores.data], dim=1)

        new_classes = []
        new_boxes = []
        new_scores = []
        gt_len = gt_classes.sum(dim=1)
        for b in range(batch_size):
            pos_len = int(gt_len[b])
            keep = box_nms(box_proposals[b], proposal_scores[b], threshold=self.nms_thr)
            neg_idx = (cls_proposals[b][keep] == 0).nonzero().squeeze()
            if len(neg_idx) > 0:
                # choose as many backgrounds as foregrounds
                random_choice = torch.randperm(len(neg_idx))[:pos_len].type(dtype).long()
                neg_idx = neg_idx[random_choice]
            else:
                neg_idx = keep.new()

            pos_idx = keep.new(list(range(pos_len)))
            keep = torch.cat([pos_idx,neg_idx],dim=0)

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


    def forward_old(self, cls_proposals, gt_classes, box_proposals, gt_boxes, proposal_scores):
        batch_size, num_boxes = gt_boxes.shape[:2]
        gt_scores = gt_classes
        # let's use only background proposals
        pos_mask = (cls_proposals > 0)
        proposal_scores[pos_mask] = 0

        cls_proposals = torch.cat([gt_classes.data, cls_proposals.data], dim=1)
        box_proposals = torch.cat([gt_boxes.data, box_proposals.data], dim=1)
        proposal_scores = torch.cat([gt_scores.data, proposal_scores.data], dim=1)

        new_classes = []
        new_boxes = []
        new_scores = []
        for b in range(batch_size):
            gt_len = int(gt_classes[b].sum())
            keep = box_nms(box_proposals[b], proposal_scores[b], threshold=self.nms_thr)
            max_nonzero_idx = proposal_scores[b][keep].nonzero().max()
            max_instances = min(max_nonzero_idx, self.max_instances)

            keep = keep[:max_instances]
            keep, _ = keep.sort()
            # make sure ground-truthes are kept
            keep[:gt_len] = keep.new(list(range(gt_len)))

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
