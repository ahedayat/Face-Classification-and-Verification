import torch
import torch.nn as nn
import torch.nn.functional as F

arc_face_eps = 1e-10


class ArcFaceLoss(nn.Module):
    """
        Implementation of ArcFace Loss Function
    """

    def __init__(self, radius=64, margin=0.5):
        super().__init__()
        self.radius = radius
        self.margin = margin

    def forward(self, w, x, y):
        """
            Calculating Loss Value
        """
        x = F.normalize(x)
        w = F.normalize(w)
        logit = x @ w

        theta = torch.arccos(torch.clip(
            logit, min=-1+arc_face_eps, max=+1-arc_face_eps))

        target_logit = torch.cos(theta+self.margin)

        logit = logit * (1-y) + target_logit * y

        logit *= self.radius

        exp_logit = torch.exp(logit)
        exp_target_logit = exp_logit * y

        loss = exp_target_logit.sum(dim=2) / exp_logit.sum(dim=2)

        loss = torch.log(loss)

        return loss.mean()
