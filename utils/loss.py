import torch
import torch.nn as nn

SMOOTH = 1e-6
bce_loss = nn.BCELoss(size_average=True)


class RMSELoss(nn.Module):
    """
    RMSE Loss Function
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,outputs,targets):
        loss = torch.sqrt(self.mse(outputs,targets) + self.eps)
        return loss

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    """
    BCE loss applied on all side outputs of the model
    """
    loss0 = bce_loss(d0,labels_v)
    loss1 = bce_loss(d1,labels_v)
    loss2 = bce_loss(d2,labels_v)
    loss3 = bce_loss(d3,labels_v)
    loss4 = bce_loss(d4,labels_v)
    loss5 = bce_loss(d5,labels_v)
    loss6 = bce_loss(d6,labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    return loss

def iou_pytorch(outputs: torch.Tensor, targets: torch.Tensor):
    """
    Calculates IOU of the two image tensors
    """
    eps = 1e-6
    outputs = outputs.cpu().data.numpy()
    targets = targets.cpu().data.numpy()
    outputs_ = outputs > 0.5
    targets_ = targets > 0.5
    intersect = (outputs_ & targets_).sum()
    union = (outputs_ | targets_).sum()
    iou = (intersect + eps)/ (union + eps)
    return iou