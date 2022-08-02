import os
import utils as utils
from datetime import datetime, date
import nets.distances as distances


import torch
import nets as nets
import deeplearning as dl
import dataloaders as data
import torch.optim as optim
from torchvision import transforms, models


def save_report(df, saving_path, backbone):
    """
        Saving Output Report Dataframe that is returned in Training
    """
    _time = datetime.now()
    hour, minute, second = _time.hour, _time.minute, _time.second

    _date = date.today()
    year, month, day = _date.year, _date.month, _date.day

    report_name = "verification_{}_{}_{}_{}_{}_{}_{}.csv".format(
        backbone, year, month, day, hour, minute, second)

    print()
    print("Report file is saving ('{}')....".format(
        os.path.join(saving_path, report_name)))

    df.to_csv(os.path.join(saving_path, report_name))


def _main(args):
    # Hardware
    device = torch.device(
        "cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")

    # CNN Backbone
    assert args.backbone in ["resnet_gray", "resnet18", "resnet34",
                             "resnet50"], "CNN backbone must be one of this items: ['resnet_gray', 'resnet18', 'resnet34', 'resnet50']"
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

    net, _ = nets.load(args.ckpt_load, net, None)
    net.train(mode=False)

    if args.gpu and torch.cuda.is_available():
        if device.type == 'cuda':
            net = net.cuda(device=device)

    # Input Size
    gray = (args.backbone == "resnet_gray")

    input_transformation = transforms.Compose([
        transforms.Resize(args.input_size),
        # transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Data Path
    # - verification
    verify_base_dir, verify_df_path = args.verify_base_dir, args.verify_df_path

    verfication_dataloader = data.VerifierDataLoader(
        base_dir=verify_base_dir, csv_path=verify_df_path, transformation=input_transformation, gray=gray)

    # Distance Metric
    assert args.distance in [
        "cosine", "euclidean"], "Distance Metrics must be one of this item: ['cosine', 'euclidean']"

    _distance = None
    if args.distance == "cosine":
        _distance = distances.CosineDist()
    elif args.distance == "euclidean":
        _distance = distances.EuclideanDist()

    # Verification
    report = dl.verification(
        net=net,
        dataloader=verfication_dataloader,
        distance=_distance,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gpu=args.gpu,
        tqbar_description="Verfication"
    )

    save_report(df=report, saving_path=args.report, backbone=args.backbone)


if __name__ == "__main__":
    args = utils.get_args()
    _main(args)
