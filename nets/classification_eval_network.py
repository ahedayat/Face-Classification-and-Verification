import torch
import torch.nn as nn


class ClassificationEvalNetwork(nn.Module):
    """
        This network is used to evaluating the model 
    """

    def __init__(self, net, similarity_metrics, training_features) -> None:
        super().__init__()
        self.net = net
        self.similarity_metrics = similarity_metrics
        self.training_features = training_features

    def forward(self, x):
        """
            Forward Propagation
        """
        feature = self.net.feature(x)

        feature = feature.unsqueeze(axis=1)

        feature = feature.repeat(1, self.training_features.shape[0], 1)

        sim = self.similarity_metrics(feature, self.training_features.repeat(
            feature.shape[0], 1, 1))

        y = torch.argmax(sim, axis=1)

        return y
