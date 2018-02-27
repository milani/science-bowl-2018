import os
import time
import pdb
import torch
import numpy as np
from model.retinanet import RetinaNet
from data.loaders import train_validation_loaders, test_loader
from trainer import Trainer

DATASET_DIR = './dataset'
BATCH_SIZE=1
MAX_SIZE=800
MIN_SIZE=500
train_dir = os.path.join(DATASET_DIR,'stage1_train')
test_dir = os.path.join(DATASET_DIR,'stage1_test')

train_loader, val_loader = train_validation_loaders(train_dir,min_size=MIN_SIZE,max_size=MAX_SIZE,validation_ratio=0.1,batch_size=BATCH_SIZE,num_workers=2,pin_memory=True, shuffle=False)
test_loader = test_loader(test_dir,batch_size=BATCH_SIZE)

model = RetinaNet()

trainer = Trainer(model)
trainer.fit(train_loader, val_loader, num_epochs=20)

#        if not batch_idx % 50:
#            print("epoch %03d | batch %03d | loss %.4f | taken %.4f / %.4f" % (epoch+1, batch_idx, train_loss/(batch_idx+1),taken/(batch_idx+1),taken2/(batch_idx+1)))

