import torch
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler

from pcdet.utils import common_utils

from .dataset import DatasetTemplate
from .kitti.kitti_dataset import KittiDataset
from .kitti.kitti_dataset_ssl import KittiDatasetSSL
from .nuscenes.nuscenes_dataset import NuScenesDataset
from .waymo.waymo_dataset import WaymoDataset
from .pandaset.pandaset_dataset import PandasetDataset
from .lyft.lyft_dataset import LyftDataset
from .kitti.kitti_semi_dataset import (KittiSemiDataset, KittiLabeledDataset, KittiUnlabeledDataset, split_kitti_semi_data)

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
    'KittiDatasetSSL': KittiDatasetSSL,
    'NuScenesDataset': NuScenesDataset,
    'WaymoDataset': WaymoDataset,
    'PandasetDataset': PandasetDataset,
    'LyftDataset': LyftDataset
}

_semi_dataset_dict = {
    'KittiDatasetSSL': {
        'PARTITION_FUNC': split_kitti_semi_data,
        'PRETRAIN': KittiSemiDataset,
        'LABELED': KittiLabeledDataset,
        'UNLABELED': KittiUnlabeledDataset,
        'TEST': KittiSemiDataset,
    }
}


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4, seed=None,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=training, sampler=sampler, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=seed)
    )

    return dataset, dataloader, sampler


def build_semi_dataloader(dataset_cfg, class_names, batch_size, repeat, dist, workers=4, root_path=None,
                          logger=None, merge_all_iters_to_one_epoch=False, seed=None):
    assert not merge_all_iters_to_one_epoch

    partition_func = _semi_dataset_dict[dataset_cfg.DATASET]['PARTITION_FUNC']
    pretrain_cls = _semi_dataset_dict[dataset_cfg.DATASET]['PRETRAIN']
    labeled_cls = _semi_dataset_dict[dataset_cfg.DATASET]['LABELED']
    unlabeled_cls = _semi_dataset_dict[dataset_cfg.DATASET]['UNLABELED']
    test_cls = _semi_dataset_dict[dataset_cfg.DATASET]['TEST']

    train_infos, labeled_infos, unlabeled_infos, test_infos = partition_func(
        dataset_cfg=dataset_cfg,
        data_splits=dataset_cfg.DATA_SPLIT,
        root_path=root_path,
        logger=logger,
    )

    def create_dataloader(dataset_cls, infos, stage, training=True, shuffle=True):
        dataset = dataset_cls(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            infos=infos,
            root_path=root_path,
            logger=logger,
            training=training,
            repeat=repeat[stage]
        )
        
        if dist:
            if training:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                rank, world_size = common_utils.get_dist_info()
                sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
        return DataLoader(
            dataset, batch_size=batch_size[stage], pin_memory=True, num_workers=workers,
            shuffle=(sampler is None) and shuffle, collate_fn=dataset.collate_batch,
            drop_last=False, sampler=sampler, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=seed)
        ), dataset, sampler

    pretrain_dataloader, pretrain_dataset, pretrain_sampler = create_dataloader(pretrain_cls, train_infos, 'pretrain')
    labeled_dataloader, labeled_dataset, labeled_sampler = create_dataloader(labeled_cls, labeled_infos, 'labeled')
    unlabeled_dataloader, unlabeled_dataset, unlabeled_sampler = create_dataloader(unlabeled_cls, unlabeled_infos, 'unlabeled')
    test_dataloader, test_dataset, test_sampler = create_dataloader(test_cls, test_infos, 'test', training=False, shuffle=False)

    datasets = {'pretrain': pretrain_dataset, 'labeled': labeled_dataset, 'unlabeled': unlabeled_dataset, 'test': test_dataset}
    dataloaders = {'pretrain': pretrain_dataloader, 'labeled': labeled_dataloader, 'unlabeled': unlabeled_dataloader, 'test': test_dataloader}
    samplers = {'pretrain': pretrain_sampler, 'labeled': labeled_sampler, 'unlabeled': unlabeled_sampler, 'test': test_sampler}

    return datasets, dataloaders, samplers


def build_unsupervised_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4,
                                  logger=None, merge_all_iters_to_one_epoch=False, seed=None):
    assert merge_all_iters_to_one_epoch is False

    train_infos, test_infos, labeled_infos, unlabeled_infos = _semi_dataset_dict[dataset_cfg.DATASET]['PARTITION_FUNC'](
        dataset_cfg=dataset_cfg,
        info_paths=dataset_cfg.INFO_PATH,
        data_splits=dataset_cfg.DATA_SPLIT,
        root_path=root_path,
        logger=logger,
    )

    unlabeled_dataset = _semi_dataset_dict[dataset_cfg.DATASET]['UNLABELED_PAIR'](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        infos=unlabeled_infos,
        root_path=root_path,
        logger=logger,
    )

    if dist:
        unlabeled_sampler = torch.utils.data.distributed.DistributedSampler(unlabeled_dataset)
    else:
        unlabeled_sampler = None
    unlabeled_dataloader = DataLoader(
        unlabeled_dataset, batch_size=batch_size['unlabeled'], pin_memory=True, num_workers=workers,
        shuffle=(unlabeled_sampler is None) and True, collate_fn=unlabeled_dataset.collate_batch,
        drop_last=False, sampler=unlabeled_sampler, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=seed)
    )

    datasets = {
        'unlabeled': unlabeled_dataset,
    }
    dataloaders = {
        'unlabeled': unlabeled_dataloader,
    }
    samplers = {
        'unlabeled': unlabeled_sampler,
    }

    return datasets, dataloaders, samplers
