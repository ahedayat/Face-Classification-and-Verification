"""
In this file, the basic function for training and evaluating `Classification Network` and `Siamese Network`
"""
from matplotlib.pyplot import axis
import pandas as pd
from soupsieve import match
from tqdm import tqdm
import losses as loss_fn
import nets as nets

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable as V

import nets as nets


def classification_train(
    net,
    train_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    device,
    epoch=1,
    batch_size=16,
    num_workers=1,
    saving_path=None,
    saving_prefix="checkpoint_",
    saving_frequency=1,
    # saving_model_every_epoch=False,
    gpu=False,
    lr_scheduler=None,
    similarity_metrics=nn.CosineSimilarity(dim=1),
    start_epoch=0,
    milestones=None,
    eval_freq=1
):
    """
    Training Classification network
    --------------------------------------------------
    Parameters:
        - net (nets.ClassificationNetwork)
            * Classification Netwrok

        - train_dataloader (dataloaders.ClassifierDataLoader)
            * Data loader for train set

        - val_dataloader (dataloaders.ClassifierDataLoader)
            * Data loader for validation set

        - optimizer (torch.optim)
            * Optimizer Algorithm

        - device (torch.device)
            * Device for training network

        - epoch (int)
            * Number of training epochs

        - batch_size (int)
            * Data loading batch size

        - num_workers (int)
    """

    train_dataloader = DataLoader(dataset=train_dataloader,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=gpu and torch.cuda.is_available(),
                                  num_workers=num_workers
                                  )

    report = pd.DataFrame(
        columns=["epoch", "train/eval", "batch_size", "loss", "acc"])

    net = net.float()

    for e in range(start_epoch, start_epoch+epoch):
        net.train()
        net.activate_arc_layer()    # I should think about this line

        if (milestones is not None) and (e in milestones):
            optimizer = nets.schedule_lr(optimizer)

        with tqdm(train_dataloader, unit="batch") as tepoch:
            for (X, Y) in tepoch:
                optimizer.zero_grad()

                tepoch.set_description(f"Training @ Epoch {e}")

                X, Y = V(X), V(Y)

                if device != 'cpu' and gpu and torch.cuda.is_available():
                    if device.type == 'cuda':
                        X, Y = X.cuda(device=device), Y.cuda(device=device)
                    elif device == 'multi':
                        X, Y = nn.DataParallel(X), nn.DataParallel(Y)

                out = net(X, Y)

                correct = (torch.argmax(out, dim=1) ==
                           torch.argmax(Y, dim=1)).sum().item()  # I should think about this line

                # I should think about below line
                accuracy = (correct / batch_size)*100

                # if isinstance(criterion, loss_fn.ArcFaceLoss):
                #     loss = criterion(net.fc.weight.data, out, Y)
                # else:
                loss = criterion(out, Y)

                current_report = pd.DataFrame({
                    "epoch": [e],
                    "train/eval": ["train"],
                    "batch_size": [Y.shape[0]],
                    "loss": [loss.item()],
                    "acc": [None]})

                report = pd.concat([report, current_report])

                accuracy = int(accuracy)

                loss.backward()
                optimizer.step()

                tepoch.set_postfix(
                    loss="{:.3f}".format(loss.item()),
                    accuracy=f"{accuracy:03d}%")

        if lr_scheduler is not None:
            lr_scheduler.step()
        # print()
        # print(f"Getting Training Features @ Epoch {e}....")
        # net.eval()
        # training_features = nets.get_training_features(
        #     net=net, train_dataloader=train_dataloader.dataset, device=device)

        # if device != 'cpu' and gpu and torch.cuda.is_available():
        #     if device.type == 'cuda':
        #         training_features = training_features.cuda(device=device)
        #     elif device == 'multi':
        #         training_features = nn.DataParallel(training_features)

        # print(f"Evaluating the model @ Epoch {e}....")

        # eval_net = nets.EvalNetwork(net=net, similarity_metrics=similarity_metrics,
        #                             training_features=training_features)

        # if device != 'cpu' and gpu and torch.cuda.is_available():
        #     if device.type == 'cuda':
        #         eval_net = eval_net.cuda(device=device)
        #     elif device == 'multi':
        #         eval_net = nn.DataParallel(eval_net)

        # val_report = classification_eval(
        #     eval_net=eval_net,
        #     dataloader=val_dataloader,
        #     criterion=criterion,
        #     device=device,
        #     batch_size=batch_size,
        #     num_workers=num_workers,
        #     gpu=gpu,
        #     tqbar_description=f"Validation @ Epoch  {e}",
        #     epoch=e
        # )

        val_report = classification_eval2(
            net=net,
            dataloader=val_dataloader,
            criterion=criterion,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            gpu=gpu,
            tqbar_description=f"Validation @ Epoch  {e}",
            epoch=e
        )

        report = pd.concat([report, val_report])

        if e % saving_frequency == 0:
            nets.save(
                file_path=saving_path,
                file_name="{}_epoch_{}".format(saving_prefix, e),
                model=net,
                optimizer=optimizer
            )

    return net, report


def _classification_eval():
    pass


def classification_eval(
    eval_net,
    dataloader,
    criterion,
    device,
    batch_size=16,
    num_workers=1,
    gpu=False,
    tqbar_description="Test",
    epoch=None
):
    """
        Evaluating the model
    """
    dataloader = DataLoader(dataset=dataloader,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=gpu and torch.cuda.is_available(),
                            num_workers=num_workers
                            )

    report = pd.DataFrame(
        columns=["epoch", "train/eval", "batch_size", "loss", "acc"])

    with tqdm(dataloader, unit="batch") as tepoch:
        for (X, Y) in tepoch:
            tepoch.set_description(tqbar_description)

            X, Y = V(X), V(Y)

            if device != 'cpu' and gpu and torch.cuda.is_available():
                if device.type == 'cuda':
                    X, Y = X.cuda(device=device), Y.cuda(device=device)
                elif device == 'multi':
                    X, Y = nn.DataParallel(X), nn.DataParallel(Y)

            output = eval_net(X)
            output2 = eval_net.net(X, Y)

            correct = (output == torch.argmax(Y, axis=1)).sum().item()
            accuracy = (correct / batch_size) * 100

            loss = criterion(output2, Y)

            current_report = pd.DataFrame({
                "epoch": [epoch],
                "train/val/test": ["val"],
                "batch_size": [Y.shape[0]],
                "loss": [loss.item()],
                "acc": [accuracy]})
            report = pd.concat([report, current_report])

            accuracy = int(accuracy)

            tepoch.set_postfix(
                loss="{:.3f}".format(loss.item()),
                accuracy=f"{accuracy:03d}%")
    return report


def classification_eval2(
    net,
    dataloader,
    criterion,
    device,
    batch_size=16,
    num_workers=1,
    gpu=False,
    tqbar_description="Test",
    epoch=None
):
    """
    Evaluation Function
    """
    dataloader = DataLoader(dataset=dataloader,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=gpu and torch.cuda.is_available(),
                            num_workers=num_workers
                            )

    report = pd.DataFrame(
        columns=["epoch", "train/eval", "batch_size", "loss", "acc"])

    net = net.float()
    net.train(mode=False)
    # net.deactivate_arc_layer()

    with tqdm(dataloader, unit="batch") as tepoch:
        for (X, Y) in tepoch:
            tepoch.set_description(tqbar_description)

            # X = torch.unsqueeze(X, dim=1).float()

            # if isinstance(criterion, nn.CrossEntropyLoss):
            #     Y = torch.tensor(Y, dtype=torch.float)
            # if isinstance(criterion, nn.NLLLoss):
            #     Y = torch.argmax(Y, dim=1)
            #     # Y = torch.tensor(Y, dtype=torch.long)

            X, Y = V(X), V(Y)

            if device != 'cpu' and gpu and torch.cuda.is_available():
                if device.type == 'cuda':
                    X, Y = X.cuda(device=device), Y.cuda(device=device)
                elif device == 'multi':
                    X, Y = nn.DataParallel(X), nn.DataParallel(Y)

            out = net(X, Y)

            correct = (torch.argmax(out, dim=1) ==
                       torch.argmax(Y, dim=1)).sum().item()
            accuracy = (correct / batch_size)*100

            # if isinstance(criterion, loss_fn.ArcFaceLoss):
            #     loss = criterion(net.fc.weight.data, out, Y)
            # else:

            loss = criterion(out, torch.argmax(Y, axis=1))

            current_report = pd.DataFrame({
                "epoch": [epoch],
                "train/eval": ["eval"],
                "batch_size": [Y.shape[0]],
                "loss": [loss.item()],
                "acc": [accuracy]})

            report = pd.concat([report, current_report])

            accuracy = int(accuracy)

            tepoch.set_postfix(
                loss="{:.3f}".format(loss.item()),
                accuracy=f"{accuracy:03d}")

    return report


# def train_features(
#     net,
#     train_dataloader,
# ):
#     """
#         Calculating Train Set Features
#     """
#     net = net.float()
#     net.train(mode=False)
#     net.deactivate_arc_layer()

#     features = None
#     for class_id, label in enumerate(train_dataloader.get_classes()):
#         class_images = train_dataloader.get_class(class_id)
#         class_features = net.feature(class_images)
#         class_features = class_features.mean(axis=0).unsqueeze(axis=0)

#         if features is None:
#             features = class_features
#         else:
#             features = torch.concat([features, class_features], axis=0)

#     return features


def verification(
        net,
        dataloader,
        distance,
        device,
        batch_size=16,
        num_workers=1,
        gpu=False,
        tqbar_description="Verification"):
    """
        Verify two images are same or different
    """

    dataloader = DataLoader(dataset=dataloader,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=gpu and torch.cuda.is_available(),
                            num_workers=num_workers
                            )

    report = pd.DataFrame(
        columns=["image_A", "image_B", "distance", "match"])

    net = net.float()
    net.train(mode=False)

    with tqdm(dataloader, unit="batch") as tepoch:
        for (X_1, X_2, file_1, file_2, match) in tepoch:
            tepoch.set_description(tqbar_description)
            

            X_1, X_2 = V(X_1), V(X_2)

            if device != 'cpu' and gpu and torch.cuda.is_available():
                if device.type == 'cuda':
                    X_1, X_2 = X_1.cuda(device=device), X_2.cuda(device=device)
                elif device == 'multi':
                    X_1, X_2 = nn.DataParallel(X_1), nn.DataParallel(X_2)
            
            
            feature_1 = net.feature(X_1)
            feature_2 = net.feature(X_2)
            dist = distance(feature_1, feature_2)

            file_1 = list(file_1)
            file_2 = list(file_2)

            current_report = pd.DataFrame({
                "image_A": file_1,
                "image_B": file_2,
                "distance": dist.tolist(),
                "match": match
            })

            # import pdb
            # pdb.set_trace()
            report = pd.concat([report, current_report])

            # tepoch.set_postfix(
            #     loss="{:.3f}".format(loss.item()),
            #     accuracy=f"{accuracy:03d}")

    return report
