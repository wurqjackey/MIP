""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler
from .lr_scheduler import WarmupMultiStepLR
import torch


def create_scheduler(cfg, optimizer):
    num_epochs = cfg.SOLVER.MAX_EPOCHS
    # type 1
    # lr_min = 0.01 * cfg.SOLVER.BASE_LR
    # warmup_lr_init = 0.001 * cfg.SOLVER.BASE_LR
    # type 2
    lr_min = cfg.SOLVER.MIN_INDEX * cfg.SOLVER.BASE_LR
    warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR
    # type 3
    # lr_min = 0.001 * cfg.SOLVER.BASE_LR
    # warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR

    warmup_t = cfg.SOLVER.WARMUP_EPOCHS
    noise_range = None
    if cfg.SOLVER.SCHEDULER == 'cosine':
        lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=num_epochs,
                lr_min=lr_min,
                t_mul= 1.,
                decay_rate=0.1,
                warmup_lr_init=warmup_lr_init,
                warmup_t=warmup_t,
                cycle_limit=1,
                t_in_epochs=True,
                noise_range_t=noise_range,
                noise_pct= 0.67,
                noise_std= 1.,
                noise_seed=42,
            )
    elif cfg.SOLVER.SCHEDULER == 'cosine-refine':
        lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=cfg.SOLVER.COSINE_EPOCHS,
                lr_min=lr_min,
                t_mul= 1.,
                decay_rate=0.1,
                warmup_lr_init=warmup_lr_init,
                warmup_t=warmup_t,
                cycle_limit=10,
                t_in_epochs=True,
                noise_range_t=noise_range,
                noise_pct= 0.67,
                noise_std= 1.,
                noise_seed=42,
            )
    else:
        lr_scheduler = WarmupMultiStepLR(
                optimizer, 
                cfg.SOLVER.STEPS, 
                cfg.SOLVER.GAMMA,
                warmup_factor = 0.01,
                warmup_iters= warmup_t,
            )
    
    return lr_scheduler
