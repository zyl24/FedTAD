import random

import numpy as np
import torch
import torch.nn as nn

from util.fgl_dataset import FGLDataset


def load_dataset(args):
    if args.dataset in [
        "Cora",
        "CiteSeer",
        "PubMed",
        "Computers",
        "Photo",
        "CS",
        "Physics",
        "NELL",
    ]:
        dataset = FGLDataset(
            args,
            root=args.root,
            name=args.dataset,
            num_clients=args.num_clients,
            partition=args.partition,
            train=0.2,
            val=0.4,
            test=0.4,
            part_delta=args.part_delta,
        )
    elif args.dataset in ["ogbn-arxiv", "Flickr"]:
        dataset = FGLDataset(
            args,
            root=args.root,
            name=args.dataset,
            num_clients=args.num_clients,
            partition=args.partition,
            train=0.6,
            val=0.2,
            test=0.2,
        )
    elif args.dataset in ["Reddit"]:
        dataset = FGLDataset(
            root=args.root,
            name=args.dataset,
            num_clients=args.num_clients,
            partition=args.partition,
            train=0.8,
            val=0.1,
            test=0.1,
        )
    elif args.dataset in ["ogbn-products"]:
        dataset = FGLDataset(
            root=args.root,
            name=args.dataset,
            num_clients=args.num_clients,
            partition=args.partition,
            train=0.1,
            val=0.05,
            test=0.85,
        )
    return dataset


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
