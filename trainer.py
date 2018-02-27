import torch
from tqdm import tqdm
from model.loss import FocalLoss
from model.anchors import Anchorizer
from torch.autograd import Variable

class Trainer(object):
    def __init__(self, model, lr=1e-4):
        super(Trainer,self).__init__()
        self.model = model
        self.anchorizer = Anchorizer()
        self.loss_fn = FocalLoss()
        self.optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.cuda = torch.cuda.is_available()
        self.loss_fn = FocalLoss()
        if self.cuda:
            model.cuda()


    def fit(self, train_loader, val_loader=None, num_epochs=20, initial_epoch=0):
        loss_fn = self.loss_fn
        model = self.model
        optimizer = self.optim
        anchorize = self.anchorizer.encode
        batch_size = train_loader.batch_size
        total_size = len(train_loader)*batch_size

        avg_loss = 0
        samples_seen = 0

        model.train()

        for epoch in range(initial_epoch, num_epochs):
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


    def evaluate(self, val_loader):
        model = self.model
        model.eval()
        anchorize = self.anchorizer.encode
        loss_fn = self.loss_fn

        avg_loss = 0
        samples_seen = 0
        for batch_idx, (imgs, masks, boxes, classes, _) in enumerate(val_loader):
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
