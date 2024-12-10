import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing, RandomErasing_ori
from .sampler import RandomIdentitySampler, RandomIdentityModalitySampler
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .occ_duke import OCC_DukeMTMCreID
from .vehicleid import VehicleID
from .veri import VeRi
from .sysu_mm import SYSU_mm
from .regdb import RegDB
from .fmy_reid import fmyreid
from .ChannelAug import *
import numpy as np

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,
    'sysu_mm': SYSU_mm,
    'regdb': RegDB,
    'fmyreid': fmyreid
}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids, _, modality_flag = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    modality_flag = torch.tensor(modality_flag, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids, modality_flag

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths, modality_flag = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    modality_flag = torch.tensor(modality_flag, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, modality_flag, img_paths

def get_sampler_(cfg, dataset):
    random.seed(cfg.DATASETS.SAMPLER_TRIAL)
    np.random.seed(cfg.DATASETS.SAMPLER_TRIAL)
    if cfg.DATASETS.SAMPLER == 'normal':
        sampler_ = RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
    elif cfg.DATASETS.SAMPLER == 'modal':
        sampler_ = RandomIdentityModalitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
    else:
        sampler_ = RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
    return sampler_

def make_dataloader(cfg):
    if cfg.INPUT.AUG == 0:
        train_transforms = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.Pad(cfg.INPUT.PADDING),
                T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
                RandomErasing_ori(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
                #T.RandomGrayscale(0.5),
                # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
            ])
    elif cfg.INPUT.AUG == 1:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.RandomChoice([T.ColorJitter(brightness=0.3,contrast=0.3),
					        T.GaussianBlur(21, sigma=(0.1, 3))]),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing_ori(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            #T.RandomGrayscale(0.5),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    elif cfg.INPUT.AUG == 20:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.RandomGrayscale(0.5),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing_ori(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            #T.RandomGrayscale(0.5),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    elif cfg.INPUT.AUG == 2:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.RandomChoice([T.ColorJitter(brightness=0.3,contrast=0.3),
					        T.GaussianBlur(21, sigma=(0.1, 3))]),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing_ori(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            T.RandomGrayscale(0.5),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    elif cfg.INPUT.AUG == 3:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.RandomChoice([T.ColorJitter(brightness=0.3,contrast=0.3),
					        T.GaussianBlur(21, sigma=(0.1, 3))]),
            T.RandomGrayscale(0.5),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing_ori(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            #T.RandomGrayscale(0.5),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    elif cfg.INPUT.AUG == 4:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.RandomChoice([T.ColorJitter(brightness=0.3,contrast=0.3),
					        T.GaussianBlur(21, sigma=(0.1, 3))]),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            ChannelRandomErasing(probability = 0.5, mean = [0.5,0.5,0.5]),
            #ChannelAdapGray(probability =0.5),
            #ChannelExchange(gray = 2),
        ])
    elif cfg.INPUT.AUG == 5:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.RandomChoice([T.ColorJitter(brightness=0.3,contrast=0.3),
					        T.GaussianBlur(21, sigma=(0.1, 3))]),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing_ori(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            #ChannelRandomErasing(probability = 0.5),
            ChannelAdapGray(probability =0.5),
            #ChannelExchange(gray = 2),
        ])
    elif cfg.INPUT.AUG == 6:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.RandomChoice([T.ColorJitter(brightness=0.3,contrast=0.3),
					        T.GaussianBlur(21, sigma=(0.1, 3))]),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            ChannelRandomErasing(probability = 0.5),
            ChannelAdapGray(probability =0.5),
            #ChannelExchange(gray = 2),
        ])
    elif cfg.INPUT.AUG == 7:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.RandomChoice([T.ColorJitter(brightness=0.3,contrast=0.3),
					        T.GaussianBlur(21, sigma=(0.1, 3))]),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            ChannelRandomErasing(probability = 0.5),
            #ChannelAdapGray(probability =0.5),
            ChannelExchange(gray = 2),
        ])
    elif cfg.INPUT.AUG == 8:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.RandomChoice([T.ColorJitter(brightness=0.3,contrast=0.3),
					        T.GaussianBlur(21, sigma=(0.1, 3))]),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            ChannelRandomErasing(probability = 0.5),
            ChannelAdapGray(probability =0.5),
            ChannelExchange(gray = 2),
        ])
    elif cfg.INPUT.AUG == 9:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.RandomChoice([T.ColorJitter(brightness=0.3,contrast=0.3),
					        T.GaussianBlur(21, sigma=(0.1, 3))]),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing_ori(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            ChannelAdapGray(probability =0.5),
            ChannelExchange(gray = 2),
        ])
    elif cfg.INPUT.AUG == 10:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.RandomChoice([T.ColorJitter(brightness=0.3,contrast=0.3),
					        T.GaussianBlur(21, sigma=(0.1, 3))]),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            ChannelRandomErasing(probability = 0.5, mean = [0.5,0.5,0.5]),
            ChannelAdapGray(probability =0.5),
            ChannelExchange(gray = 2),
        ])
    elif cfg.INPUT.AUG == 11:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.RandomChoice([T.ColorJitter(brightness=0.3,contrast=0.3),
					        T.GaussianBlur(21, sigma=(0.1, 3))]),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing_ori(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            ChannelAdapGray(probability =0.5),
            ChannelExchange(gray = 2),
        ])
    elif cfg.INPUT.AUG == 12:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.RandomChoice([T.ColorJitter(brightness=0.3,contrast=0.3),
					        T.GaussianBlur(21, sigma=(0.1, 3))]),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing_ori(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            ChannelAdapGray(probability =0.5),
        ])
    elif cfg.INPUT.AUG == 13: 
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.RandomGrayscale(0.5),
            T.RandomChoice([T.ColorJitter(brightness=0.3,contrast=0.3),
					        T.GaussianBlur(21, sigma=(0.1, 3))]),
            T.ToTensor(),
            RandomizedQuantizationAugModule(region_num=8, transforms_like=True),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing_ori(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ])
    elif cfg.INPUT.AUG == 14: 
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            #T.RandomGrayscale(0.5),
            T.RandomChoice([T.ColorJitter(brightness=0.3,contrast=0.3),
					        T.GaussianBlur(21, sigma=(0.1, 3))]),
            T.ToTensor(),
            RandomizedQuantizationAugModule(region_num=8, transforms_like=True),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing_ori(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ])
    elif cfg.INPUT.AUG == 15: 
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            #T.RandomGrayscale(0.5),
            #T.RandomChoice([T.ColorJitter(brightness=0.3,contrast=0.3),
			#		        T.GaussianBlur(21, sigma=(0.1, 3))]),
            T.ToTensor(),
            RandomizedQuantizationAugModule(region_num=8, transforms_like=True),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing_ori(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ])
    elif cfg.INPUT.AUG == 16: 
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.RandomGrayscale(0.5),
            T.RandomChoice([T.ColorJitter(brightness=0.3,contrast=0.3),
					        T.GaussianBlur(21, sigma=(0.1, 3))]),
            T.RandomSolarize(100, p=0.5),
            #RandomizedQuantizationAugModule(region_num=8, transforms_like=True),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing_ori(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ])
    elif cfg.INPUT.AUG == 17: 
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.RandomGrayscale(0.5),
            T.RandomChoice([T.ColorJitter(brightness=0.3,contrast=0.3),
					        T.GaussianBlur(21, sigma=(0.1, 3))]),
            T.RandomSolarize(100, p=0.5),
            T.ToTensor(),
            RandomizedQuantizationAugModule(region_num=8, transforms_like=True),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing_ori(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ])
    elif cfg.INPUT.AUG == 18: 
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.RandomGrayscale(0.5),
            #T.RandomChoice([T.ColorJitter(brightness=0.3,contrast=0.3),
			#		        T.GaussianBlur(21, sigma=(0.1, 3))]),
            T.ToTensor(),
            RandomizedQuantizationAugModule(region_num=8, transforms_like=True),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing_ori(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ])
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            sampler_ = get_sampler_(cfg, dataset)
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                #sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                sampler=sampler_,
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num
