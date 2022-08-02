import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .net_utils import l2_norm


class ArcFaceLayer(nn.Module):
    """
        Arc Face Last Layer
    """

    def __init__(self, in_features=512, out_features=1000, s=64, m=0.5, _device=torch.device("cpu")) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.s = s
        self.m = m
        self._device = _device

        self.weights = Parameter(torch.Tensor(
            self.in_features, self.out_features))
        # nn.init.xavier_uniform_(self.weights)
        self.weights.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        # self.mm = math.sin(math.pi - m) * m
        self.mm = self.sin_m * m

    def forward(self, embbedings, label):
        """
            Forward propagation
        """
        # cosine = F.linear(F.normalize(input), F.normalize(self.weights))
        # sine = torch.sqrt((1 - torch.pow(cosine, 2)).clamp(0, 1))

        # cos_phi = cosine * self.cos_m - sine * self.sin_m

        # cos_phi = torch.where(cosine > self.th, cos_phi, cosine - self.mm)

        # # one_hot = torch.zeros(cosine.size(), device=self._device)
        # # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # one_hot = label

        # output = (one_hot * cos_phi) + ((1-one_hot) * cosine)
        # output *= self.s

        # return output

# weights norm

        nB = len(embbedings)
        weights_norm = l2_norm(self.weights, axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings, weights_norm)
#         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.th
        cond_mask = cond_v <= 0
        # when theta not in [0,pi], use cosface instead
        keep_val = (cos_theta - self.mm)
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        # a little bit hacky way to prevent in_place operation on cos_theta
        output = cos_theta * 1.0
        idx_ = torch.arange(0, nB, dtype=torch.long)
        label = torch.argmax(label, axis=1)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output
