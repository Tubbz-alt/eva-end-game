import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import gc
import time
from loss import *

test_losses = []
test_acc = []


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
        gc.collect()
        running_loss = 0
        miou = 0.0
        mrmse = 0.0
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
            miou += iou_pytorch(seg_out, mask)
            rmse = RMSELoss()
            mrmse += rmse(dep_out, dense_depth)

            del dep_out,d2,d3,d4,d5,d6,d7,seg_out,s2,s3,s4,s5,s6,s7,loss
            gc.collect()
            end_time = time.time()
            pbar.set_description(desc=f'Batch_id={batch_idx} Test set: Average loss: {running_loss / ite_num4val} Accuracy IOU(Segmentation)={miou/ite_num4val} RMSE(Dense depth)={mrmse/ite_num4val}  Avg Batch Time={end_time-start_time} Secs)')
            





