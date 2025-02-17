import os
import random
import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import CMPDataset

#保证多进程数据加载复现一致性
def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def create_dataloader(opt):
    #shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = CMPDataset(opt)
    
    g = torch.Generator()
    g.manual_seed(0)
    #sampler = get_bal_sampler(dataset) if opt.class_bal else None
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt['batch_size'],
                                              shuffle=True,
                                              sampler=None,
                                              num_workers=int(opt['num_threads']),
                                              worker_init_fn=seed_worker,
                                              generator=g)
    return data_loader
