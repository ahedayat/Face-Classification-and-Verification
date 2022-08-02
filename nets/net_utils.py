import os
import torch
from tqdm import tqdm


def save_net(file_path, file_name, model, optimizer=None):
    """
        In this function, a model is saved.
        ------------------------------------------------
        Parameters:
            - file_path (str): saving path
            - file_name (str): saving name
            - model (torch.nn.Module)
            - optimizer (torch.optim)
    """
    state_dict = dict()

    state_dict["model"] = model.state_dict()

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()

    torch.save(state_dict, os.path.join(file_path, file_name))


def load_net(ckpt_path, model, optimizer=None):
    print("Loading the model from '{}'".format(ckpt_path))
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model"])
    if (optimizer is not None) and ("optimizer" in checkpoint.keys()):
        optimizer.load_state_dict(checkpoint["optimizer"])

    return model, optimizer


def get_training_features(net, train_dataloader, device):
    """
        Return training featurs :
            - output: (num_class, embedding_size)

    """
    labels = train_dataloader.get_classes()

    training_features = None

    counter = 0

    with tqdm(labels) as t_labels:
        for ix, label in enumerate(t_labels):
            counter += 1
            data_label = train_dataloader.data_class(label)
            data_label = data_label.unsqueeze(dim=1)

            if device != 'cpu' and torch.cuda.is_available():
                if device.type == 'cuda':
                    data_label = data_label.cuda(device=device)
                elif device == 'multi':
                    data_label = nn.DataParallel(data_label)

            feature = net.feature(data_label)
            feature = feature.mean(axis=0).unsqueeze(dim=0)

            if training_features is None:
                training_features = feature
            else:
                training_features = torch.concat([training_features, feature])

            if counter == 10:
                torch.save(training_features,
                           "./features/{}_{}.pt".format(ix-9, ix+1))
                counter = 0
                del training_features
                training_features = None

    print("Concatinating Training Features...")
    training_features = None

    for file_name in os.listdir("./features/"):
        feature = torch.load(f"./features/{file_name}")

        if training_features is None:
            training_features = feature
        else:
            training_features = torch.concat([training_features, feature])

    return training_features


def l2_norm(input, axis=1):
    """
        returns normalized tensor
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def schedule_lr(optimizer):
    """
        Learning Rate Decay (1/10)
    """
    for params in optimizer.param_groups:
        params['lr'] /= 10
    return optimizer
