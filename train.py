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
MAX_SIZE=320
MIN_SIZE=320
NUM_WORKERS=2
NUM_EPOCHS=250
BASE_LR=10e-4

train_dir = os.path.join(DATASET_DIR,'stage1_train')

train_loader, val_loader = train_validation_loaders(train_dir,min_size=MIN_SIZE,max_size=MAX_SIZE,validation_ratio=0.1,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=True, shuffle=True)

model = RetinaNet()

trainer = Trainer(model, force_single_gpu=False, lr=BASE_LR)

try:
    trainer.load_checkpoint(trainer.latest_checkpoint())
except RuntimeError as err:
    print("Warn: No checkpoints loaded.")
    print(err)

trainer.fit(train_loader, val_loader, num_epochs=NUM_EPOCHS)
