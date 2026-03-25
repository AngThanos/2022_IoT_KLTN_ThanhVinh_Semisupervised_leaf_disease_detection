import copy
import logging
import math
import os
import os.path
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from . import augmentations as img_trsform
from .base import BaseDataset

# https://pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id):
    cur_seed = np.random.get_state()[1][0]
    cur_seed = int(cur_seed) + worker_id
    np.random.seed(cur_seed)
    random.seed(cur_seed)


def yolo_collate_fn(batch):
    """Custom collate function for YOLO dataset with variable-length labels.
    
    Args:
        batch: list of tuples (index, image, label) or (index, image_weak, image_strong, label)
    
    Returns:
        Batched data with labels containing batch indices
    """
    # Determine if semi-supervised (4 items) or supervised (3 items)
    if len(batch[0]) == 3:
        indices, images, labels = zip(*batch)
        indices = torch.tensor(indices)
        images = torch.stack(images, dim=0)
        
        # Process labels: add batch index to each label
        new_labels = []
        for batch_idx, label in enumerate(labels):
            if len(label) > 0:
                batch_indices = torch.full((len(label), 1), batch_idx, dtype=label.dtype)
                new_labels.append(torch.cat([batch_indices, label], dim=1))
        
        if new_labels:
            labels = torch.cat(new_labels, dim=0)
        else:
            labels = torch.zeros((0, 6))
        
        return indices, images, labels
    else:
        # Semi-supervised: (index, image_weak, image_strong, label)
        indices, images_weak, images_strong, labels = zip(*batch)
        indices = torch.tensor(indices)
        images_weak = torch.stack(images_weak, dim=0)
        images_strong = torch.stack(images_strong, dim=0)
        
        # Process labels: add batch index to each label
        new_labels = []
        for batch_idx, label in enumerate(labels):
            if len(label) > 0:
                batch_indices = torch.full((len(label), 1), batch_idx, dtype=label.dtype)
                new_labels.append(torch.cat([batch_indices, label], dim=1))
        
        if new_labels:
            labels = torch.cat(new_labels, dim=0)
        else:
            labels = torch.zeros((0, 6))
        
        return indices, images_weak, images_strong, labels


class yolo_dset(BaseDataset):
    def __init__(self, data_root, data_list, trs_form, trs_form_strong=None, 
        seed=0, n_sup=5866, split="val", flag_semi=False,
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ):
        super(yolo_dset, self).__init__(data_list)
        self.data_root = data_root
        self.transform_weak = trs_form
        self.transform_strong = trs_form_strong
        self.flag_semi = flag_semi
        self.split = split
        # random.seed(seed)

        self.trf_normalize = self._get_to_tensor_and_normalize(mean, std)

        # oversamplying labeled data for semi-supervised training
        if len(self.list_sample) >= n_sup and split == "train":
            self.list_sample_new = random.sample(self.list_sample, n_sup)
        elif len(self.list_sample) < n_sup and split == "train":
            num_repeat = math.ceil(n_sup / len(self.list_sample))
            self.list_sample = self.list_sample * num_repeat

            self.list_sample_new = random.sample(self.list_sample, n_sup)
        else:
            self.list_sample_new = self.list_sample

    def _load_valid_sample(self, index, max_retry=20):
        """Try loading a sample, skipping corrupted images by moving to the next index."""
        logger = logging.getLogger("global")
        data_len = len(self.list_sample_new)
        if data_len == 0:
            raise RuntimeError("Dataset is empty.")

        max_retry = min(max_retry, data_len)
        for offset in range(max_retry):
            cur_index = (index + offset) % data_len
            image_path = os.path.join(self.data_root, self.list_sample_new[cur_index][0])
            label_path = os.path.join(self.data_root, self.list_sample_new[cur_index][1])
            try:
                image = self.img_loader(image_path, "RGB")
                label = self.label_loader(label_path)
                if offset > 0:
                    logger.warning(
                        f"Recovered sample loading by skipping to index {cur_index} after {offset} retries"
                    )
                return cur_index, image, label
            except Exception as e:
                logger.warning(f"Skipping unreadable image: {image_path} ({e})")

        raise RuntimeError(
            f"Failed to load a valid image after {max_retry} retries starting from index {index}."
        )

    @staticmethod
    def _get_to_tensor_and_normalize(mean, std):
        return img_trsform.ToTensorAndNormalize(mean, std)

    def __getitem__(self, index):
        # load image and its label
        sample_index, image, label = self._load_valid_sample(index)

        if self.transform_strong is None:
            image, label = self.transform_weak(image, label)
            # print(image.shape, label.shape)
            image, label = self.trf_normalize(image, label)
            if not self.flag_semi:
                return sample_index, image, label
            else:
                return sample_index, image, image.clone(), label
        else:
            # apply augmentation
            image_weak, label = self.transform_weak(image, label)
            image_strong = self.transform_strong(image_weak)
            # print("="*100)
            # print(index, image_weak.size, image_strong.size, label.size)
            # print("="*100)

            image_weak, label = self.trf_normalize(image_weak, label)
            image_strong, _ = self.trf_normalize(image_strong, label)
            # print(index, image_weak.shape, image_strong.shape,label.shape)

            return sample_index, image_weak, image_strong, label

        # image, label = self.transform(image, label)
        # return image[0], label[0, 0].long()

    def __len__(self):
        return len(self.list_sample_new)


def build_additional_strong_transform(cfg):
    assert cfg.get("strong_aug", False) != False
    strong_aug_nums = cfg["strong_aug"].get("num_augs", 2)
    flag_use_rand_num = cfg["strong_aug"].get("flag_use_random_num_sampling", True)
    strong_img_aug = img_trsform.strong_img_aug(strong_aug_nums,
            flag_using_random_num=flag_use_rand_num)
    return strong_img_aug


def build_basic_transfrom(cfg, split="val", mean=[0.485, 0.456, 0.406]):
    ignore_label = cfg["ignore_label"]
    trs_form = []
    
    # Always resize images to a consistent size
    base_size = cfg.get("resize_base_size", 640)
    if isinstance(base_size, int):
        base_size = [base_size, base_size]
    
    if split != "val":
        if cfg.get("rand_resize", False):
            trs_form.append(img_trsform.Resize(base_size, cfg["rand_resize"]))
        
        if cfg.get("flip", False):
            trs_form.append(img_trsform.RandomFlip(prob=0.5, flag_hflip=True))
    
        # crop also sometime for validating
        if cfg.get("crop", False):
            crop_size, crop_type = cfg["crop"]["size"], cfg["crop"]["type"]
            trs_form.append(img_trsform.Crop(crop_size, crop_type=crop_type, mean=mean, ignore_value=ignore_label))
    else:
        # For validation, resize to fixed size without random scaling
        trs_form.append(img_trsform.Resize(base_size, [1.0, 1.0]))

    return img_trsform.Compose(trs_form)


def build_yololoader(split, all_cfg, seed=0):
    # extract augs config from "train"/"val" into the higher level.
    cfg_dset = all_cfg["dataset"]
    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    # set up workers and batchsize
    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = cfg.get("n_sup", 5866)

    # build transform
    mean, std = cfg["mean"], cfg["std"]
    trs_form = build_basic_transfrom(cfg, split=split, mean=mean)

    # create dataset
    dset = yolo_dset(cfg["data_root"], cfg["data_list"], trs_form, None, 
        seed, n_sup, mean=mean, std=std)

    # build sampler
    sample = DistributedSampler(dset)
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=workers,
        sampler=sample,
        shuffle=False,
        pin_memory=False,
        worker_init_fn=seed_worker,
        collate_fn=yolo_collate_fn,
    )
    return loader


def build_semi_yololoader(split, all_cfg, seed=0):
    split = "train"
    # extract augs config from "train" into the higher level.
    cfg_dset = all_cfg["dataset"]
    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    # set up workers and batchsize
    workers = cfg.get("workers", 2) 
    batch_size = cfg.get("batch_size", 2)
    # n_sup is the number of supervised samples to use directly.
    n_sup = 5866 - cfg.get("n_sup", 5866)

    # build transform
    mean, std = cfg["mean"], cfg["std"]
    trs_form_weak = build_basic_transfrom(cfg, split=split, mean=mean)
    if cfg.get("strong_aug", False):
        trs_form_strong = build_additional_strong_transform(cfg)
    else:
        trs_form_strong = None

    dset = yolo_dset(cfg["data_root"], cfg["data_list"], trs_form_weak, None, 
                    seed, n_sup, split=split, mean=mean, std=std)

    sample_sup = DistributedSampler(dset)

    data_list_unsup = cfg["data_list"].replace("/labeled/", "/unlabeled/")
    dset_unsup = yolo_dset(cfg["data_root"], data_list_unsup, trs_form_weak, trs_form_strong,
                            seed, n_sup, split,
                            flag_semi=True,
                            mean=mean, std=std)
    sample_unsup = DistributedSampler(dset_unsup)

    # create dataloader
    loader_sup = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=workers,
        sampler=sample_sup,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        collate_fn=yolo_collate_fn,
    )
    loader_unsup = DataLoader(
        dset_unsup,
        batch_size=batch_size,
        num_workers=workers,
        sampler=sample_unsup,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        collate_fn=yolo_collate_fn,
    )
    return loader_sup, loader_unsup
