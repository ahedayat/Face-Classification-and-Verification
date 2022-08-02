
import os
import nets as nets
import utils as utils
import dataloaders as data
import deeplearning as dl
from datetime import datetime, date

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader


def save_report(df, backbone_name, saving_path):
    """
        Saving Output Report Dataframe that is returned in Training
    """
    _time = datetime.now()
    hour, minute, second = _time.hour, _time.minute, _time.second

    _date = date.today()
    year, month, day = _date.year, _date.month, _date.day

    report_name = "{}_{}_{}_{}_{}_{}_{}.csv".format(
        backbone_name, year, month, day, hour, minute, second)

    df.to_csv(os.path.join(saving_path, report_name))


def _main(args):
    # Hardware
    gpu = args.gpu
    device = torch.device(
        "cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")

    # Data Loaders
    # - Train
    train_input_transformation = transforms.Compose([
        transforms.Resize(args.input_size),
        # transforms.RandomCrop(args.input_size - 20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    train_base_dir, train_df_path = args.train_base_dir, args.train_df_path
    train_dataloader = data.ClassifierDataLoader(
        train_base_dir, train_df_path, train_input_transformation, gray=True)
    train_dataloader = DataLoader(dataset=train_dataloader,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=gpu and torch.cuda.is_available(),
                                  num_workers=args.num_workers
                                  )

    # - Validation
    eval_input_transformation = transforms.Compose([
        transforms.Resize(args.input_size),
        # transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    val_base_dir, val_df_path = args.val_base_dir, args.val_df_path
    val_dataloader = data.ClassifierDataLoader(
        val_base_dir, val_df_path, eval_input_transformation, gray=True)

    # Loss Function
    criterion = torch.nn.CrossEntropyLoss()

    # Network
    resnet = nets.resnet_face18()
    m = 0.5
    s = 30
    net = nets.TrainNetwork(
        num_cats=1000,
        resnet=resnet,
        s=s,
        m=m,
        device_last_layer=device
    )

    if args.gpu and torch.cuda.is_available():
        if device.type == 'cuda':
            net = net.cuda(device=device)

    # optimizer
    optimizer = optim.Adam([{'params': net.resnet.parameters()}, {'params': net.last_layer.parameters()}],
                           lr=args.lr, weight_decay=5e-4)
    # Load the Model
    ckpt_path = args.ckpt_path
    net, optimizer = nets.load(
        ckpt_path=ckpt_path,
        model=net,
        optimizer=optimizer
    )
    if device != 'cpu' and gpu and torch.cuda.is_available():
        if device.type == 'cuda':
            net = net.cuda(device=device)
        elif device == 'multi':
            net = nn.DataParallel(net)
    optimizer = optim.Adam([{'params': net.resnet.parameters()}, {'params': net.last_layer.parameters()}],
                           lr=args.lr, weight_decay=5e-4)

    # Getting Training Features
    net.eval()

    training_features = nets.get_training_features(
        net=net, train_dataloader=train_dataloader.dataset, device=device)

    if device != 'cpu' and gpu and torch.cuda.is_available():
        if device.type == 'cuda':
            training_features = training_features.cuda(device=device)
        elif device == 'multi':
            training_features = nn.DataParallel(training_features)

    # Evaluating the model
    similarity_metrics = nn.CosineSimilarity(dim=1)
    eval_net = nets.EvalNetwork(net=net, similarity_metrics=similarity_metrics,
                                training_features=training_features)

    if device != 'cpu' and gpu and torch.cuda.is_available():
        if device.type == 'cuda':
            eval_net = eval_net.cuda(device=device)
        elif device == 'multi':
            eval_net = nn.DataParallel(eval_net)

    val_report = dl.classification_eval(
        eval_net=eval_net,
        dataloader=val_dataloader,
        criterion=criterion,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gpu=gpu,
        tqbar_description=f"Validation @ {ckpt_path}",
        epoch=45
    )

    save_report(df=val_report, backbone_name="val_resnet_gray_".format(ckpt_path.split("/")[-1]),
                saving_path=args.report)


if __name__ == "__main__":
    args = utils.get_args()
    _main(args)
