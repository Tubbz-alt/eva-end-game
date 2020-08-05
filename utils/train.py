import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.autograd import Variable
import time
import gc

train_losses = []
train_acc = []
running_loss = 0.0
bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

    loss0 = bce_loss(d0,labels_v)
    loss1 = bce_loss(d1,labels_v)
    loss2 = bce_loss(d2,labels_v)
    loss3 = bce_loss(d3,labels_v)
    loss4 = bce_loss(d4,labels_v)
    loss5 = bce_loss(d5,labels_v)
    loss6 = bce_loss(d6,labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    return loss

def getOptimizer(model, lr=0.001, momentum=0.9, weight_decay=0):
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
  return optimizer
  
def train(model, device, train_loader, optimizer, depth_criterion, seg_criterion, l1_factor=0, scheduler=None):
  ite_num4val = 0
  # save_frq = 2000 # save the model every 2000 iterations

  model.train()
  pbar = tqdm(train_loader)
  running_loss = 0.0
  torch.cuda.empty_cache()
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
      seg_loss = seg_criterion(seg_out, s1, s2, s3, s4, s5, s6, mask)
      
      loss = dep_loss + seg_loss
      running_loss += loss.item()

      # Backpropagation
      loss.backward()
      optimizer.step()
      del seg_out, s1, s2, s3, s4, s5, s6, depth_out, d1, d2, d3, d4, d5, d6, dep_loss, seg_loss, loss
      gc.collect()

      if scheduler:
        scheduler.step()

      end_time = time.time()
      pbar.set_description(desc= f'Loss={running_loss / ite_num4val}  Batch_id={batch_idx} Total Time={end_time-start_time} Secs')
      