import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import gc
import time

test_losses = []
test_acc = []
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

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def __test(model, device, test_loader, depth_criterion, seg_criterion):
    model.eval()
    ite_num4val = 0
    with torch.no_grad():
        pbar = tqdm(test_loader)
        torch.cuda.empty_cache()
        running_loss = 0
        for batch_idx, data in enumerate(pbar):
            start_time = time.time()
            ite_num4val = ite_num4val + 1

            bg, fg_bg, dense_depth, mask = data['bg'], data['fg_bg'], data['dense_depth'], data['fg_bg_mask']
            bg = bg.to(device=device, dtype=torch.float32)
            fg_bg = fg_bg.to(device=device, dtype=torch.float32)
            dense_depth = dense_depth.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.float32)

            inputs = torch.cat([bg,fg_bg], dim=1)

            dep_out,d2,d3,d4,d5,d6,d7,seg_out,s2,s3,s4,s5,s6,s7 = model(inputs)
            dep_loss = depth_criterion(dep_out, d2, d3, d4, d5, d6, d7, dense_depth)
            seg_loss = seg_criterion(seg_out, s2, s3, s4, s5, s6, s7, mask)
            
            loss = dep_loss + seg_loss
      
            running_loss += loss.item()

            del dep_out,d2,d3,d4,d5,d6,d7,seg_out,s2,s3,s4,s5,s6,s7,loss
            gc.collect()
            end_time = time.time()
            pbar.set_description(desc=f'Batch_id={batch_idx} Test set: Average loss: {running_loss / ite_num4val}  Batch Time={end_time-start_time} Secs)')
            





