import torch.nn as nn
import torch
import torch.nn.functional as F
from PIL import Image
from utils import *
import numpy as np


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def predict(model, device, data):
    """
    Runs the inference on the data and returns the result
    """
    model.eval()
    with torch.no_grad():
        bg, fg_bg, dense_depth, mask = data['bg'], data['fg_bg'], data['dense_depth'], data['fg_bg_mask']
        bg = bg.to(device=device, dtype=torch.float32)
        fg_bg = fg_bg.to(device=device, dtype=torch.float32)
        dense_depth = dense_depth.to(device=device, dtype=torch.float32)
        mask = mask.to(device=device, dtype=torch.float32)
        inputs = torch.cat([bg,fg_bg], dim=1)
        dep_out,d2,d3,d4,d5,d6,d7,seg_out,s2,s3,s4,s5,s6,s7 = model(inputs)
        dep_pred = dep_out[:,0,:,:]
        dep_pred = normPRED(dep_pred)
        seg_pred = seg_out[:,0,:,:]
        seg_pred = normPRED(seg_pred)
        del dep_out,d2,d3,d4,d5,d6,d7,seg_out,s2,s3,s4,s5,s6,s7
        dep_predict = dep_pred.squeeze()
        dep_predict_np = dep_predict.cpu().data.numpy()
        dep_predict_np = (dep_predict_np*255).astype(np.uint8)
        seg_predict = seg_pred.squeeze()
        seg_predict_np = seg_predict.cpu().data.numpy()
        seg_predict_np = (seg_predict_np*255).astype(np.uint8)

        return dep_predict_np, seg_predict_np
