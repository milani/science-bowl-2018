import os
import torch
import torch.nn.functional as F
import numpy as np
import pdb
from collections import OrderedDict
from glob import glob
from tqdm import tqdm
from model.utils import box_iou, mask_iou
from torch.autograd import Variable

class Trainer(object):
    def __init__(self, model, checkpointing=True, log_dir='./checkpoints', lr=1e-4, force_single_gpu=False):
        super(Trainer,self).__init__()
        params = filter(lambda x: x.requires_grad, model.parameters())
        self.optim = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
        self.cuda = torch.cuda.is_available()
        self.initial_epoch = 1
        self.epoch = 1
        self.checkpointing = checkpointing
        self.log_dir = log_dir

        if not force_single_gpu and torch.cuda.device_count() > 1:
            print("Using %d GPUs. You will only benefit if batch_size > 1 and num_workers in data loaders > 2." % torch.cuda.device_count())
            model = torch.nn.DataParallel(model)
        if self.cuda:
            model.cuda()
        self.model = model


    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.initial_epoch = checkpoint['epoch']

        try:
            self.model.load_state_dict(checkpoint['state_dict'])
        except KeyError as e:
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)

        self.optim.load_state_dict(checkpoint['optimizer'])


    def save_checkpoint(self, path):
        checkpoint = {
                'epoch': self.epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optim.state_dict()
        }
        torch.save(checkpoint, path)


    def checkpoint(self):
        if self.checkpointing:

            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            path = os.path.join(self.log_dir,'%d.npy' % self.epoch)
            self.save_checkpoint(path)


    def latest_checkpoint(self):
        if not os.path.exists(self.log_dir):
            raise RuntimeError('log_dir does not exist')

        paths = glob(os.path.join(self.log_dir,'*.npy'))
        epochs = []
        for path in paths:
            epochs.append(int(path.split('/')[-1].replace('.npy','')))

        if len(epochs) == 0:
            raise RuntimeError('no checkpoints available')

        epochs.sort()
        latest = epochs[-1]
        return os.path.join(self.log_dir, '%d.npy' % latest)


    def fit(self, train_loader, val_loader=None, num_epochs=20, lr=None):
        if self.initial_epoch >= num_epochs:
            print("Already trained for %d epochs. Requested %d." % (self.initial_epoch, num_epochs))
            return
        model = self.model
        optimizer = self.optim
        batch_size = train_loader.batch_size
        total_size = len(train_loader)*batch_size

        if isinstance(lr, float):
            optimizer.lr = lr

        model.train()

        for epoch in range(self.initial_epoch, num_epochs + 1):
            avg_loss = 0
            avg_cls_loss = 0
            avg_box_loss = 0
            avg_mask_loss = 0

            samples_seen = 0

            pbar = tqdm(total=total_size, leave=True, unit='b')
            for batch_idx, (imgs, masks, boxes, classes, _) in enumerate(train_loader):
                batch_size = imgs.shape[0]

                if self.cuda:
                    imgs = imgs.cuda()
                    masks = masks.cuda()
                    boxes = boxes.cuda()
                    classes = classes.cuda()

                imgs = Variable(imgs)
                classes = Variable(classes, requires_grad=False)
                boxes = Variable(boxes, requires_grad=False)
                masks = Variable(masks, requires_grad=False)

                _, _, _, cls_loss, box_loss, mask_loss, total_loss = model(imgs, classes, boxes, masks)

                samples_seen += batch_size
                optimizer.zero_grad()
                total_loss.mean().backward()
                optimizer.step()

                avg_loss = (samples_seen * avg_loss + batch_size * total_loss.data[0]) / (samples_seen + batch_size)
                avg_cls_loss = (samples_seen * avg_cls_loss + batch_size * cls_loss.data[0]) / (samples_seen + batch_size)
                avg_box_loss = (samples_seen * avg_box_loss + batch_size * box_loss.data[0]) / (samples_seen + batch_size)
                avg_mask_loss = (samples_seen * avg_mask_loss + batch_size * mask_loss.data[0]) / (samples_seen + batch_size)
                pbar.update(batch_size)
                pbar.set_description('Epoch %d/%d' % (epoch, num_epochs))
                pbar.set_postfix({'loss':avg_loss, 'class loss':avg_cls_loss,'box loss':avg_box_loss, 'mask loss':avg_mask_loss})

            if val_loader is not None:
                eval_loss, box_metric, mask_metric = self.evaluate(val_loader)

            pbar.set_postfix({'loss':avg_loss, 'val_loss':eval_loss, 'box mAP':box_metric, 'mask mAP':mask_metric})
            pbar.close()
            self.epoch = epoch
            self.checkpoint()


    def evaluate(self, val_loader):
        model = self.model
        model.eval()

        avg_loss = 0
        box_metric, mask_metric = 0, 0
        samples_seen = 0
        for batch_idx, (imgs, masks, boxes, classes, names) in enumerate(val_loader):
            batch_size = imgs.shape[0]
            input_size = imgs.shape[2:]

            if self.cuda:
                imgs = imgs.cuda()
                masks = masks.cuda()
                boxes = boxes.cuda()
                classes = classes.cuda()

            imgs = Variable(imgs, volatile=True)
            masks = Variable(masks, volatile=True)
            boxes = Variable(boxes, volatile=True)
            classes = Variable(classes, volatile=True)

            returns = model(imgs, classes, boxes, masks)
            cls_proposals, box_proposals, mask_preds, cls_loss, box_loss, mask_loss, total_loss = returns
            avg_loss = (samples_seen * avg_loss + batch_size * total_loss.data[0]) / (samples_seen + batch_size)

            box_m, mask_m = self.evaluate_metrics(cls_proposals, classes, box_proposals, boxes, mask_preds, masks, input_size)
            box_metric, mask_metric = box_metric + box_m, mask_metric + mask_m

            samples_seen +=  batch_size


        model.train()
        return avg_loss, box_metric/samples_seen, mask_metric/samples_seen


    def evaluate_metrics(self, cls_preds, classes, box_preds, boxes, mask_preds, masks, input_size):
        box_prec = self.evaluate_box_precision(cls_preds, classes, box_preds, boxes, input_size)
        mask_prec = self.evaluate_mask_precision(mask_preds, masks)
        return box_prec, mask_prec

    def evaluate_box_precision(self, cls_preds, classes, box_preds, boxes, input_size):
        ious = box_iou(box_preds.data, boxes.data, order='xyxy')
        num_pred = (cls_preds.data > 0).sum(dim=1).float()
        num_true = (classes.data > 0).sum(dim=1).float()
        p = 0
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        # in case of multiple detection for the same object, one detection is considered
        # as true positive, the others are false detections.
        for t in thresholds:
            tp = (ious > t).sum(dim=1).clamp(max=1).sum(dim=1).float()
            fp = num_pred - tp
            fn = num_true - tp
            p += tp / (tp + fp + fn)

        return p.sum() / len(thresholds)


    def evaluate_mask_precision(self, mask_preds, masks):
        ious = mask_iou(mask_preds, masks)
        thresholds = list(np.arange(0.5, 1.0, 0.05))
        p = 0
        for iou in ious:
            for t in thresholds:
                matches = iou > t
                tp = np.sum(np.sum(matches, axis=1) == 1)
                fp = np.sum(np.sum(matches, axis=0) == 0)
                fn = np.sum(np.sum(matches, axis=1) == 0)
                p += tp / (tp + fp + fn)
        n = len(thresholds) * len(ious)
        return p / n


    def predict(self, loader):
        model = self.model
        model.predict()
        batch_size = loader.batch_size
        total_size = len(loader)*batch_size

        pbar = tqdm(total=total_size, leave=True, unit='batches')
        classes = []
        boxes = []
        masks = []
        for batch_idx, (imgs,_) in enumerate(loader):
            batch_size = imgs.shape[0]

            if self.cuda:
                imgs = imgs.cuda()

            imgs = Variable(imgs, volatile=True)
            b_classes, b_boxes, b_masks = model(imgs)
            classes.append(b_classes)
            boxes.append(b_boxes)
            masks.append(b_masks)

            pbar.update(batch_size)
        pbar.close()
        return classes, boxes, masks


    def predict_on_batch(self, imgs):
        model = self.model
        model.predict()
        batch_size = imgs.shape[0]

        if self.cuda:
            imgs = imgs.cuda()

        imgs = Variable(imgs, volatile=True)
        return model(imgs)
