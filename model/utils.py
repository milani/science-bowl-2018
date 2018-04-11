"""
From https://github.com/kuangliu/pytorch-retinanet/blob/master/utils.py
"""
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from skimage.transform import resize
from lib.nms.pth_nms import pth_nms

def meshgrid(x, y, row_major=True):
    a = torch.arange(0,x)
    b = torch.arange(0,y)
    xx = a.repeat(y).view(-1,1)
    yy = b.view(-1,1).repeat(1,x).view(-1,1)
    return torch.cat([xx,yy],1) if row_major else torch.cat([yy,xx],1)

def place_masks(masks, boxes, input_size):
    masks = masks.data.sigmoid()
    boxes = boxes.data.round().long()
    in_height, in_width = input_size
    boxes[:,:,0].clamp_(min=0)
    boxes[:,:,1].clamp_(min=0)
    boxes[:,:,2].clamp_(max=in_width - 1)
    boxes[:,:,3].clamp_(max=in_height -1)

    output = []
    for bmask, bbox in zip(masks, boxes):
        rmask = []
        for i, box in enumerate(bbox):
            mask = bmask[i]
            resized_mask = masks.new(in_height,in_width).fill_(0)

            x1, y1, x2, y2 = box
            height, width = y2 - y1 + 1, x2 - x1 + 1

            if height > 1 and width > 1:
                scaled_mask = torch.Tensor(resize(mask,(height,width),mode='constant'))
                resized_mask[y1:(y2+1), x1:(x2+1)] = scaled_mask > 0.5

            rmask.append(resized_mask.unsqueeze(0))
        output.append(torch.cat(rmask, 0).unsqueeze(0))
    return Variable(torch.cat(output,0))

def seq_label(masks):
    # num_batch x num_masks x height x width
    num_batch, num_masks = masks.shape[0:2]
    labels = masks.new(list(range(num_masks))) + 1
    labels = labels.unsqueeze(0).expand(num_batch,-1).unsqueeze(2).unsqueeze(2)
    masks = torch.sum(masks * labels,dim=1)
    return masks

def mask_iou(mask_preds, masks):
    masks = seq_label(masks.data)
    mask_preds = seq_label(mask_preds.data)
    masks = masks.cpu().numpy()
    mask_preds = mask_preds.cpu().numpy()
    batch_size = masks.shape[0]

    ious = []
    for b in range(batch_size):
        true = masks[b]
        pred = mask_preds[b]
        true_objects = int(true.max()) + 1
        pred_objects = int(pred.max()) + 1
        intersection = np.histogram2d(true.flatten(), pred.flatten(), bins=(true_objects, pred_objects))[0]
        area_true = np.histogram(true, bins=true_objects)[0]
        area_pred = np.histogram(pred, bins=pred_objects)[0]
        area_true = np.expand_dims(area_true, -1)
        area_pred = np.expand_dims(area_pred, 0)
        union = area_true + area_pred - intersection

        intersection = intersection[1:,1:]
        union = union[1:,1:]
        union[union == 0] = 1e-9

        ious.append(intersection/union)

    return ious

def change_box_order(boxes, order):
    '''Change box order between (xmin,ymin,xmax,ymax) and (xcenter,ycenter,width,height).
    Args:
      boxes: (tensor) bounding boxes, sized [N,4].
      order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.
    Returns:
      (tensor) converted bounding boxes, sized [N,4].
    '''
    assert order in ['xyxy2xywh','xywh2xyxy']
    dim = len(boxes.shape)
    if dim == 2:
        a = boxes[:, :2]
        b = boxes[:, 2:]
    else:
        a = boxes[:, :, :2]
        b = boxes[:, :, 2:]
    if order == 'xyxy2xywh':
        return torch.cat([(a+b)/2,b-a], dim - 1)
    return torch.cat([a-b/2,a+b/2], dim - 1)

def box_iou(box1, box2, order='xyxy'):
    '''Compute the intersection over union of two set of boxes.
    The default box order is (xmin, ymin, xmax, ymax).
    Args:
      box1: (tensor) bounding boxes, sized [b, N,4].
      box2: (tensor) bounding boxes, sized [b, M,4].
      order: (str) box order, either 'xyxy' or 'xywh'.
    Return:
      (tensor) iou, sized [b, N,M].
    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    if order == 'xywh':
        box1 = change_box_order(box1, 'xywh2xyxy')
        box2 = change_box_order(box2, 'xywh2xyxy')

    N = box1.size(1)
    M = box2.size(1)

    # To avoid memory peak, I do some operations in-place
    # and re-order some others.

    area1 = (box1[:,:,2]-box1[:,:,0]) * (box1[:,:,3]-box1[:,:,1])  # [N,]
    area2 = (box2[:,:,2]-box2[:,:,0]) * (box2[:,:,3]-box2[:,:,1])  # [M,]

    box1 = box1[:,:,None,:]
    box2 = box2[:,None,:,:]
    # right_bottom - left_top + 1
    wh = torch.min(box1[:,:,:,2:], box2[:,:,:,2:])
    wh.sub_(torch.max(box1[:,:,:,:2], box2[:,:,:,:2]))
    wh.clamp_(min=0)

    inter = wh[:,:,:,0] * wh[:,:,:,1]  # [b, N,M]
    del wh

    union = (area1[:,:,None] + area2[:,None,:] - inter)
    del area1
    del area2

    iou = inter / union
    # avoid nan
    iou[iou != iou] = 0
    return iou

def box_nms(bboxes, scores, threshold=0.5, mode='union'):
    dets = torch.cat([bboxes, scores.unsqueeze(1)], dim=1)
    return pth_nms(dets, threshold)

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels.long()]     # [N,D]

def crop_masks(masks, boxes, max_pool=False, pooling_size=21):
    """
    [  x2-x1             x1 + x2 - W + 1  ]
    [  -----      0      ---------------  ]
    [  W - 1                  W - 1       ]
    [                                     ]
    [           y2-y1    y1 + y2 - H + 1  ]
    [    0      -----    ---------------  ]
    [           H - 1         H - 1      ]
    """
    crops = []
    for mask,box in zip(masks.data,boxes.data):

        mask = mask.unsqueeze(1)

        x1 = box[:, 0]
        y1 = box[:, 1]
        x2 = box[:, 2]
        y2 = box[:, 3]

        height = mask.size(2)
        width = mask.size(3)

        # affine theta
        theta = torch.autograd.Variable(box.new(box.size(0), 2, 3).zero_())
        theta[:, 0, 0] = ((x2 - x1) / (width - 1)).view(-1)
        theta[:, 0 ,2] = ((x1 + x2 - width + 1) / (width - 1)).view(-1)
        theta[:, 1, 1] = ((y2 - y1) / (height - 1)).view(-1)
        theta[:, 1, 2] = ((y1 + y2 - height + 1) / (height - 1)).view(-1)

        pre_pool_size = pooling_size * 2 if max_pool else pooling_size
        grid = F.affine_grid(theta, torch.Size((box.size(0), 1, pre_pool_size, pre_pool_size)))
        crop = F.grid_sample(mask, grid)
        if max_pool:
            crop = F.max_pool2d(crop, 2, 2)
        crops.append(crop.permute(1,0,2,3))

    return torch.cat(crops,0)

