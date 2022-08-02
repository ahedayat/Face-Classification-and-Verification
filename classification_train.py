import os
import utils as utils
import losses as loss
from nets import ArcFaceLayer
from datetime import datetime, date


import torch
import nets as nets
import deeplearning as dl
import dataloaders as data
import torch.optim as optim
from torchvision import transforms, models
from torch.optim.lr_scheduler import StepLR


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
    device = torch.device(
        "cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")

    # CNN Backbone
    assert args.backbone in ["resnet_gray", "resnet18", "resnet34",
                             "resnet50"], "CNN backbone must be one of this items: ['resnet18', 'resnet34', 'resnet50']"
    resnet = None
    if args.backbone == "resnet_gray":
        resnet = nets.resnet_face18()
    elif args.backbone == "resnet18":
        resnet = models.resnet18(pretrained=args.pretrained, progress=True)
    elif args.backbone == "resnet34":
        resnet = models.resnet34(pretrained=args.pretrained, progress=True)
    else:
        resnet = models.resnet50(pretrained=args.pretrained, progress=True)

    # ArcFace Parameters
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

    # Input Image Features
    gray = (args.backbone == "resnet_gray")

    train_input_transformation = transforms.Compose([
        transforms.Resize(args.input_size),
        # transforms.RandomCrop(args.input_size - 20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # eval_input_transformation = train_input_transformation

    eval_input_transformation = transforms.Compose([
        transforms.Resize(args.input_size),
        # transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Data Path
    # - Train
    train_base_dir, train_df_path = args.train_base_dir, args.train_df_path
    train_dataloader = data.ClassifierDataLoader(
        train_base_dir, train_df_path, train_input_transformation, gray=gray)

    # - Validation
    val_base_dir, val_df_path = args.val_base_dir, args.val_df_path
    val_dataloader = data.ClassifierDataLoader(
        val_base_dir, val_df_path, eval_input_transformation, gray=gray)

    # Loss Function
    assert args.criterion in [
        'arcface', 'cross_entropy'], "Loss Function must be one of this items: ['arcface', 'cross_entropy']"

    # if args.criterion == "arcface":
    #     margin = args.arcface_m
    #     radius = args.arcface_s
    #     criterion = loss.ArcFaceLoss(radius=radius, margin=margin)
    # else:
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    assert args.optimizer in [
        "sgd", "adam"], "Optimizer must be one of this items: ['sgd', 'adam']"

    if args.optimizer == "sgd":
        # optimizer = optim.SGD(net.parameters(), lr=args.lr,
        #                       momentum=0.9, weight_decay=5e-4)
        optimizer = optim.SGD([{'params': net.resnet.parameters()}, {'params': net.last_layer.parameters()}],
                              lr=args.lr, momentum=0.9, weight_decay=5e-4)

    else:
        # optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
        optimizer = optim.Adam([{'params': net.resnet.parameters()}, {'params': net.last_layer.parameters()}],
                               lr=args.lr, weight_decay=5e-4)

    start_epoch = 0

    if args.ckpt_load is not None:
        print("Loading `{}`...".format(args.ckpt_load))
        net, optimizer = nets.load(
            ckpt_path=args.ckpt_load,
            model=net,
            optimizer=optimizer
        )
        start_epoch = args.ckpt_load.split("_")[-1]
        start_epoch = int(start_epoch)+1

    # Learning Rate Schedular
    lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Checkpoint Address
    saving_path, saving_prefix = args.ckpt_path, args.ckpt_prefix
    saving_frequency = args.save_freq

    # Training
    net, report = dl.classification_train(
        net=net,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epoch=args.epoch,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        saving_path=saving_path,
        saving_prefix=saving_prefix,
        saving_frequency=saving_frequency,
        # saving_model_every_epoch=False,
        gpu=args.gpu,
        lr_scheduler=lr_scheduler,
        start_epoch=start_epoch,
        milestones=[20, 30, 40, 60, 90])

    save_report(df=report, backbone_name=args.backbone,
                saving_path=args.report)

    nets.save(
        file_path=saving_path,
        file_name="{}_final".format(saving_prefix),
        model=net,
        optimizer=optimizer
    )


if __name__ == "__main__":
    args = utils.get_args()
    _main(args)
