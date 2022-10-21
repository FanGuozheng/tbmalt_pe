#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 12:12:51 2022

@author: gz_fan
"""
from torch.utils.data import DataLoader as _DataLoader
from tbmalt.io.dataset import Dataset


class DataLoader(_DataLoader):
    """An interface to load data inherited from `torch.utils.data.DataLoader`.

    Arguments:
        dataset:
        batch_size:
        shuffle:
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        generator=None,
        prefetch_factor=None,
        persistent_workers=None):
    """

    def __init__(
            self,
            dataset: Dataset,
            batch_size: int,
            shuffle: bool = False,
            sampler=None,
            batch_sampler=None,
            num_workers=0,
            collate_fn=None,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            generator=None,
            prefetch_factor: int = 2,
            persistent_workers=None):
        super(DataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers)

    def __getitem__(self):
        pass

    @property
    def _index_sampler(self):
        pass

    def pin_memory(self):
        pass

    def collate_wrapper(self):
        pass

    def __len__(self) -> int:
        pass
