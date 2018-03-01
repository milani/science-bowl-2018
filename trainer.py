import torch
import pdb
import os
from glob import glob
from tqdm import tqdm
from model.loss import FocalLoss
from model.anchors import Anchorizer
from model.utils import box_iou
from torch.autograd import Variable

class Trainer(object):
    def __init__(self, model, checkpointing=True, log_dir='./checkpoints', lr=1e-4):
        super(Trainer,self).__init__()
        self.model = model
        self.anchorizer = Anchorizer()
        self.loss_fn = FocalLoss()
        self.optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.cuda = torch.cuda.is_available()
        self.loss_fn = FocalLoss()
        self.initial_epoch = 1
        self.epoch = 1
        self.checkpointing = checkpointing
        self.log_dir = log_dir
        if self.cuda:
            model.cuda()


    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.initial_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
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
        epochs.sort()
        latest = epochs[-1]
        return os.path.join(self.log_dir, '%d.npy' % latest)


    def fit(self, train_loader, val_loader=None, num_epochs=20):
        loss_fn = self.loss_fn
        model = self.model
        optimizer = self.optim
        anchorize = self.anchorizer.encode
        batch_size = train_loader.batch_size
        total_size = len(train_loader)*batch_size

        model.train()

        for epoch in range(self.initial_epoch, num_epochs + 1):
            avg_loss = 0
            samples_seen = 0

            pbar = tqdm(total=total_size, leave=True, unit='batches')
            for batch_idx, (imgs, masks, boxes, classes, _) in enumerate(train_loader):
                batch_size = imgs.shape[0]

                if self.cuda:
                    boxes = boxes.cuda()
                    classes = classes.cuda()

                classes, boxes = anchorize(classes, boxes, imgs.shape[2:])

                if self.cuda:
                    imgs = imgs.cuda()

                imgs = Variable(imgs)
                boxes = Variable(boxes, requires_grad=False)
                classes = Variable(classes, requires_grad=False)

                cls_preds, box_preds = model(imgs)

                loss = loss_fn(cls_preds, classes, box_preds, boxes)
                samples_seen += batch_size
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_loss = (samples_seen * avg_loss + batch_size * loss.data[0]) / (samples_seen + batch_size)
                pbar.update(batch_size)
                pbar.set_description('Epoch %d/%d' % (epoch, num_epochs))
                pbar.set_postfix({'loss':avg_loss})

            if val_loader is not None:
                eval_loss = self.evaluate(val_loader)

            pbar.set_postfix({'loss':avg_loss, 'val_loss':eval_loss})
            pbar.close()
            self.epoch = epoch
            self.checkpoint()


    def evaluate(self, val_loader):
        model = self.model
        model.eval()
        anchorize = self.anchorizer.encode
        loss_fn = self.loss_fn

        avg_loss = 0
        samples_seen = 0
        for batch_idx, (imgs, masks, boxes, classes, names) in enumerate(val_loader):
            batch_size = imgs.shape[0]

            if self.cuda:
                boxes = boxes.cuda()
                classes = classes.cuda()

            classes, boxes = anchorize(classes, boxes, imgs.shape[2:])

            if self.cuda:
                imgs = imgs.cuda()

            imgs = Variable(imgs, requires_grad=False)
            boxes = Variable(boxes, requires_grad=False)
            classes = Variable(classes, requires_grad=False)

            cls_preds, box_preds = model(imgs)
            loss = loss_fn(cls_preds, classes, box_preds, boxes)

            samples_seen +=  batch_size
            avg_loss = (samples_seen * avg_loss + batch_size * loss.data[0]) / (samples_seen + batch_size)


        model.train()
        return avg_loss


    def evaluate_metric(self,cls_preds, classes, box_preds, boxes, input_size):
        deanchorize = self.anchorizer.decode
        cls_preds, box_preds = deanchorize(cls_preds.data, box_preds.data,input_size)
        #classes, boxes = deanchorize(classes.data.unsqueeze(2), boxes.data, input_size)
        results = 0
        for b in range(len(cls_preds)):
            print(type(box_preds[b]))
            print(type(boxes[b]))
            ious = box_iou(box_preds[b].unsqueeze(0), boxes[b].unsqueeze(0), order='xyxy').squeeze(0)
            num_pred = len(cls_preds[b])
            num_true = classes[b].nonzero().squeeze().shape[0]
            p = 0
            for t in [0.5,0.6,0.7,0.8,0.9]:
                matches = (ious > t).nonzero()
                tp = len(matches)
                fp = num_pred - tp
                fn = num_true - tp
                p += tp / (tp + fp + fn)
            results += p/5
        return results/len(cls_preds)


    def predict(self, loader):
        model = self.model
        model.eval()
        deanchorize = self.anchorizer.decode
        batch_size = loader.batch_size
        total_size = len(loader)*batch_size

        pbar = tqdm(total=total_size, leave=True, unit='batches')
        classes = []
        boxes = []
        for batch_idx, (imgs,_) in enumerate(loader):
            batch_size = imgs.shape[0]

            if self.cuda:
                imgs = imgs.cuda()

            imgs = Variable(imgs, requires_grad=False)
            cls_preds, box_preds = model(imgs)
            b_classes, b_boxes = deanchorize(cls_preds.data, box_preds.data, imgs.shape[2:])
            classes.append(b_classes)
            boxes.append(b_boxes)

            pbar.update(batch_size)
        pbar.close()
        return classes, boxes


    def predict_on_batch(self, imgs):
        model = self.model
        model.eval()
        deanchorize = self.anchorizer.decode
        batch_size = imgs.shape[0]

        if self.cuda:
            imgs = imgs.cuda()

        imgs = Variable(imgs, requires_grad=False)
        cls_preds, box_preds = model(imgs)
        classes, boxes = deanchorize(cls_preds.data, box_preds.data, imgs.shape[2:])
        return classes, boxes
