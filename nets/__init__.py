"""
Supported Networks:
    - EmbeddingNetwork -> for embedding an image
"""

from .classification_train_network import ClassificationTrainNetwork as TrainNetwork
from .classification_eval_network import ClassificationEvalNetwork as EvalNetwork
from .last_layers import ArcFaceLayer
from .net_utils import save_net as save
from .net_utils import load_net as load
from .net_utils import get_training_features, schedule_lr
from .resnet_gray import *
# import .distances as distances

__version__ = '1.0.0'
__author__ = 'Ali Hedayatnia, M.Sc. Student of Artificial Intelligence @ University of Tehran'
