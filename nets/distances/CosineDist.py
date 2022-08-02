import torch
import torch.nn as nn


class CosineDist(nn.Module):
    """
        Cosine Distance Module:
            Cosine-Distance(x_1, x_2) =  (x_1 . x_2) / (||x_1||_2 * ||x_2||_2)
    """

    def __init__(self):
        super().__init__()
        self.p = 2

        self.dist_fn = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x1, x2):
        """
            This function calculate Cosine distance between two tensors.

            Parameters
            ---------------------------------------------------
                - x1 (torch.tensor)
                - x2 (torch.tensor)
        """
        dist = self.dist_fn(x1, x2)

        dist = torch.abs(dist)

        return dist
