import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.autograd import Variable
import time
import gc
from loss import *
# from pytorch_ssim import *

train_losses = []
train_acc = []
running_loss = 0.0


def getOptimizer(model, lr=0.001, momentum=0.9, weight_decay=0):
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
  return optimizer
  
def train(model, device, train_loader, optimizer, depth_criterion, seg_criterion, l1_factor=0, scheduler=None):
  ite_num4val = 0
  # save_frq = 2000 # save the model every 2000 iterations

  model.train()
  pbar = tqdm(train_loader)
  running_loss = 0.0
  miou = 0.0
  mrmse = 0.0
  torch.cuda.empty_cache()
  gc.collect()
  for batch_idx, data in enumerate(pbar):
      start_time = time.time()
      ite_num4val = ite_num4val + 1
      bg, fg_bg, dense_depth, mask = data['bg'], data['fg_bg'], data['dense_depth'], data['fg_bg_mask']

      bg = bg.to(device=device, dtype=torch.float32)
      fg_bg = fg_bg.to(device=device, dtype=torch.float32)
      dense_depth = dense_depth.to(device=device, dtype=torch.float32)
      mask = mask.to(device=device, dtype=torch.float32)

      inputs = torch.cat([bg, fg_bg], dim=1)

      optimizer.zero_grad()
      # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
      # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

      # Predict
      depth_out, d1, d2, d3, d4, d5, d6, seg_out, s1,s2,s3,s4,s5,s6 = model(inputs)

      # Calculate losslabels
      dep_loss = depth_criterion(depth_out, d1, d2, d3, d4, d5, d6, dense_depth)
      # ssim_loss = SSIM()
      # dep_loss = -ssim_loss(depth_out, dense_depth).data.item()
      seg_loss = seg_criterion(seg_out, s1, s2, s3, s4, s5, s6, mask)
      
      loss = dep_loss + seg_loss
      running_loss += loss.item()
      miou += iou_pytorch(seg_out, mask)
      rmse = RMSELoss()
      mrmse += rmse(depth_out, dense_depth)

      # Backpropagation
      loss.backward()
      optimizer.step()
      del seg_out, s1, s2, s3, s4, s5, s6, depth_out, d1, d2, d3, d4, d5, d6, dep_loss, seg_loss, loss
      gc.collect()

      if scheduler:
        scheduler.step()

      end_time = time.time()
      pbar.set_description(desc= f'Batch_id={batch_idx} Train set: Loss={running_loss / ite_num4val} Accuracy IOU(Segmentation)={miou/ite_num4val} RMSE(Dense depth)={mrmse/ite_num4val}  Avg Batch Time={end_time-start_time} Secs')
      