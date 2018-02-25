"""
From https://github.com/kuangliu/pytorch-retinanet/blob/master/utils.py
"""
import torch

def meshgrid(x, y, row_major=True):
    a = torch.arange(0,x)
    b = torch.arange(0,y)
    xx = a.repeat(y).view(-1,1)
    yy = b.view(-1,1).repeat(1,x).view(-1,1)
    return torch.cat([xx,yy],1) if row_major else torch.cat([yy,xx],1)

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

    lt = torch.max(box1[:,:,None,:2], box2[:,:,:2])  # [b, N,M,2]
    rb = torch.min(box1[:,:,None,2:], box2[:,:,2:])  # [b, N,M,2]

    wh = (rb-lt+1).clamp(min=0)      # [b, N,M,2]
    inter = wh[:,:,:,0] * wh[:,:,:,1]  # [b, N,M]

    area1 = (box1[:,:,2]-box1[:,:,0]+1) * (box1[:,:,3]-box1[:,:,1]+1)  # [N,]
    area2 = (box2[:,:,2]-box2[:,:,0]+1) * (box2[:,:,3]-box2[:,:,1]+1)  # [M,]
    iou = inter / (area1[:,:,None] + area2 - inter)
    return iou

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels.long()]            # [N,D]

