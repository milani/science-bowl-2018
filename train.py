import os
import torch
import numpy as np
from model.retinanet import RetinaNet
from model.anchors import Mapper
from loss import FocalLoss
from torch.autograd import Variable
from torch.optim import SGD
from data.loaders import train_validation_loaders, test_loader

DATASET_DIR = './dataset'
BATCH_SIZE=1
MAX_SIZE=800
MIN_SIZE=500
train_dir = os.path.join(DATASET_DIR,'stage1_train')
test_dir = os.path.join(DATASET_DIR,'stage1_test')

train_loader, validation_loader = train_validation_loaders(train_dir,min_size=MIN_SIZE,max_size=MAX_SIZE,validation_ratio=0.1,batch_size=BATCH_SIZE,num_workers=2,pin_memory=True, shuffle=False)
test_loader = test_loader(test_dir,batch_size=BATCH_SIZE)

def back_hook(module, grad_in, grad_out):
    for g in grad_in:
        print('grad_in shape:', g.data.norm())
    return None


model = RetinaNet()
#model.register_backward_hook(back_hook)

if torch.cuda.is_available():
    model.cuda()

mapper = Mapper()
cost_fn = FocalLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)

num_epochs = 40
for epoch in range(num_epochs):
    for batch_idx, (imgs, masks, boxes1, classes1, names) in enumerate(train_loader):
        classes, boxes = mapper.encode(classes1, boxes1, (MAX_SIZE, MAX_SIZE))
        imgs = Variable(imgs)
        boxes = Variable(boxes)
        classes = Variable(classes.long())

        if torch.cuda.is_available():
            imgs = imgs.cuda()
            boxes = boxes.cuda()
            classes = classes.cuda()

        try:
            cls_preds, box_preds = model(imgs)
    
            cost = cost_fn(cls_preds, classes, box_preds, boxes)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
        except:
            print("Except")
            print(names)
            print(classes1)
            print(classes)
            print(boxes1)
            print(boxes)
            raise

        if not batch_idx % 50:
            print("epoch %03d | batch %03d | cost %.4f" % (epoch+1, batch_idx, cost.data[0]))

