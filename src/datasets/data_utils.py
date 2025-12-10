import os
import re
from itertools import repeat

from hydra.utils import instantiate

from src.datasets.collate import collate_factory
from src.utils.init_utils import set_worker_seed

def normalize_text(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    return text.strip()
    

def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms[transform_type]
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_dataloaders(config, device, acoustic_model):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        device (str): device to use for batch transforms.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    # dataloaders init
    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        dataset = instantiate(config.datasets.get(dataset_partition), acoustic_model=acoustic_model)
        print(f"Size of {dataset_partition}: {len(dataset)}")

        assert dataset_partition == "train" or config.dataloader["inference"].batch_size <= len(
            dataset
        ), (
            f"The batch size ({config.dataloader['inference'].batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )
        dataloader_config = (
            config.dataloader["train"]
            if dataset_partition == "train"
            else config.dataloader["inference"]
        )
        max_workers = os.cpu_count()
        partition_dataloader = instantiate(
            dataloader_config,
            dataset=dataset,
            collate_fn=collate_factory(config),
            drop_last=(dataset_partition == "train"),
            shuffle=(dataset_partition == "train"),
            worker_init_fn=set_worker_seed,
            num_workers=max_workers,
        )

        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms
