import os
import glob
import random
import h5py
import itertools
import numpy as np
from scipy import ndimage
from scipy.ndimage.interpolation import zoom

import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import cv2


class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                          1. Samplers
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
class TwoStreamBatchSampler(Sampler): #
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        2. Generators
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample


class WeakStrongAugment(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image_org = sample["image"].copy()
        image, label = sample["image"], sample["label"]
        
        # geometry
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        # resize
        image_org = self.resize(image_org)
        image = self.resize(image)
        label = self.resize(label)
        
        # strong augmentation is color jitter
        image_strong = func_strong_augs(image, p_color=0.8, p_blur=0.2)

        # fix dimensions
        image_org = torch.from_numpy(image_org.astype(np.float32)).unsqueeze(0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image_org,
            "image_weak": image,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
    

class WeakStrongAugmentMore(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image_org = sample["image"].copy()
        image, label = sample["image"], sample["label"]
        
        # geometry
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        # elif random.random() > 0.5:
        #     image, label = random_rotate(image, label)
        
        # resize
        image_org = self.resize(image_org)
        image = self.resize(image)
        label = self.resize(label)
        
        # strong augmentation is color jitter
        image_strong = func_strong_augs(image, p_color=0.5, p_blur=0.2)
        image_strong_more = func_strong_augs(image, p_color=1.0, p_blur=0.2)

        # fix dimensions
        image_org = torch.from_numpy(image_org.astype(np.float32)).unsqueeze(0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image_org,
            "image_weak": image,
            "image_strong": image_strong,
            "image_strong_more": image_strong_more,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                         3. Augmentations
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def color_jitter(image, p=1.0):
    # if not torch.is_tensor(image):
    #     np_to_tensor = transforms.ToTensor()
    #     image = np_to_tensor(image)
    # s is the strength of color distortion.
    # s = 1.0
    # jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    jitter = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)
    if np.random.random() < p:
        image = jitter(image)
    return image


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def func_strong_augs(image, p_color=0.8, p_blur=0.5):
    img = Image.fromarray((image * 255).astype(np.uint8))
    img = color_jitter(img, p_color)
    img = blur(img, p_blur)

    img = torch.from_numpy(np.array(img)).unsqueeze(0).float() / 255.0

    return img


# ================================================================= #
#             Detection Dataset (YOLO format bounding boxes)
# ================================================================= #

def load_yolo_labels(label_path):
    """Load YOLO format labels: class_id cx cy w h (normalized).
    Returns numpy array of shape (N, 5) or empty (0, 5).
    """
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    boxes.append([cls_id, cx, cy, w, h])
    if len(boxes) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array(boxes, dtype=np.float32)


def yolo_to_xyxy(boxes, img_w, img_h):
    """Convert YOLO (cls, cx, cy, w, h) normalized -> (cls, x1, y1, x2, y2) in pixels."""
    if len(boxes) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    result = boxes.copy()
    result[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * img_w  # x1
    result[:, 2] = (boxes[:, 2] - boxes[:, 4] / 2) * img_h  # y1
    result[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * img_w  # x2
    result[:, 4] = (boxes[:, 2] + boxes[:, 4] / 2) * img_h  # y2
    return result


def xyxy_to_yolo(boxes, img_w, img_h):
    """Convert (cls, x1, y1, x2, y2) in pixels -> YOLO (cls, cx, cy, w, h) normalized."""
    if len(boxes) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    result = boxes.copy()
    result[:, 1] = ((boxes[:, 1] + boxes[:, 3]) / 2) / img_w  # cx
    result[:, 2] = ((boxes[:, 2] + boxes[:, 4]) / 2) / img_h  # cy
    result[:, 3] = (boxes[:, 3] - boxes[:, 1]) / img_w         # w
    result[:, 4] = (boxes[:, 4] - boxes[:, 2]) / img_h         # h
    return result


class DetectionDataSet(Dataset):
    """Dataset for semi-supervised object detection with YOLO-format labels.
    
    Directory structure:
        base_dir/
        ├── train/
        │   ├── labeled/
        │   │   ├── images/
        │   │   ├── labels/
        │   │   └── list.txt
        │   └── unlabeled/
        │       ├── images/
        │       └── list.txt
        └── val/
            ├── images/
            ├── labels/
            └── list.txt
    """

    def __init__(self, base_dir, split="train", transform=None, img_size=640):
        self._base_dir = base_dir
        self.split = split
        self.transform = transform
        self.img_size = img_size
        self.sample_list = []   # list of (image_path, label_path_or_None, is_labeled)

        if split == "train":
            # labeled
            labeled_dir = os.path.join(base_dir, "train", "labeled")
            with open(os.path.join(labeled_dir, "list.txt"), "r") as f:
                labeled_names = [l.strip() for l in f if l.strip()]
            for name in labeled_names:
                img_path = os.path.join(labeled_dir, "images", name + ".JPG")
                lbl_path = os.path.join(labeled_dir, "labels", name + ".txt")
                if not os.path.exists(img_path):
                    # try lowercase extension
                    img_path = os.path.join(labeled_dir, "images", name + ".jpg")
                self.sample_list.append((img_path, lbl_path, True))

            # unlabeled
            unlabeled_dir = os.path.join(base_dir, "train", "unlabeled")
            with open(os.path.join(unlabeled_dir, "list.txt"), "r") as f:
                unlabeled_names = [l.strip() for l in f if l.strip()]
            for name in unlabeled_names:
                img_path = os.path.join(unlabeled_dir, "images", name + ".JPG")
                if not os.path.exists(img_path):
                    img_path = os.path.join(unlabeled_dir, "images", name + ".jpg")
                self.sample_list.append((img_path, None, False))

            self.num_labeled = len(labeled_names)
            self.num_unlabeled = len(unlabeled_names)

        elif split == "val":
            val_dir = os.path.join(base_dir, "val")
            with open(os.path.join(val_dir, "list.txt"), "r") as f:
                val_names = [l.strip() for l in f if l.strip()]
            for name in val_names:
                img_path = os.path.join(val_dir, "images", name + ".JPG")
                lbl_path = os.path.join(val_dir, "labels", name + ".txt")
                if not os.path.exists(img_path):
                    img_path = os.path.join(val_dir, "images", name + ".jpg")
                self.sample_list.append((img_path, lbl_path, True))

        print("total {} samples ({} split)".format(len(self.sample_list), split))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        img_path, lbl_path, is_labeled = self.sample_list[idx]

        # Load image (BGR -> RGB)
        img = cv2.imread(img_path)
        assert img is not None, f"Failed to load image: {img_path}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ori_h, ori_w = img.shape[:2]

        # Load labels
        if lbl_path is not None and os.path.exists(lbl_path):
            boxes = load_yolo_labels(lbl_path)  # (N, 5) cls,cx,cy,w,h normalized
        else:
            boxes = np.zeros((0, 5), dtype=np.float32)

        sample = {"image": img, "boxes": boxes, "is_labeled": is_labeled,
                  "ori_size": (ori_h, ori_w), "idx": idx, "img_path": img_path}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class DetectionResizeTransform:
    """Resize image to fixed size and convert to tensor. For val/test."""
    def __init__(self, img_size=640):
        self.img_size = img_size

    def __call__(self, sample):
        img = sample["image"]
        boxes = sample["boxes"]

        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        # HWC -> CHW, normalize to [0,1]
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)

        # boxes are already normalized, no need to adjust
        boxes = torch.from_numpy(boxes)

        sample["image"] = img
        sample["boxes"] = boxes
        return sample


class DetectionWeakStrongAugment:
    """Weak + strong augmentations for semi-supervised detection training."""
    def __init__(self, img_size=640, p_color=0.8, p_blur=0.2):
        self.img_size = img_size
        self.p_color = p_color
        self.p_blur = p_blur

    def __call__(self, sample):
        img = sample["image"]  # (H, W, 3) uint8 RGB
        boxes = sample["boxes"]  # (N, 5) cls,cx,cy,w,h normalized

        # Keep original (resized only)
        img_org = cv2.resize(img, (self.img_size, self.img_size))

        # --- Weak augmentation: random horizontal flip ---
        img_weak = img.copy()
        boxes_weak = boxes.copy()
        if random.random() > 0.5:
            img_weak = np.fliplr(img_weak).copy()
            if len(boxes_weak) > 0:
                boxes_weak[:, 1] = 1.0 - boxes_weak[:, 1]  # flip cx

        if random.random() > 0.5:
            img_weak = np.flipud(img_weak).copy()
            if len(boxes_weak) > 0:
                boxes_weak[:, 2] = 1.0 - boxes_weak[:, 2]  # flip cy

        # Resize
        img_weak = cv2.resize(img_weak, (self.img_size, self.img_size))

        # --- Strong augmentation: color jitter + blur on weak ---
        img_strong = Image.fromarray(img_weak)
        img_strong = color_jitter(img_strong, self.p_color)
        img_strong = blur(img_strong, self.p_blur)
        img_strong = np.array(img_strong)

        # Convert to tensors (CHW, float32, [0,1])
        img_org = torch.from_numpy(img_org.astype(np.float32) / 255.0).permute(2, 0, 1)
        img_weak = torch.from_numpy(img_weak.astype(np.float32) / 255.0).permute(2, 0, 1)
        img_strong = torch.from_numpy(img_strong.astype(np.float32) / 255.0).permute(2, 0, 1)

        boxes_weak = torch.from_numpy(boxes_weak)

        sample["image"] = img_org
        sample["image_weak"] = img_weak
        sample["image_strong"] = img_strong
        sample["boxes"] = boxes_weak  # augmented boxes
        return sample


def detection_collate_fn(batch):
    """Custom collate for detection: images stack, boxes as list."""
    images = torch.stack([s["image"] for s in batch], dim=0)
    boxes = [s["boxes"] for s in batch]
    is_labeled = [s["is_labeled"] for s in batch]
    idx = [s["idx"] for s in batch]

    result = {"image": images, "boxes": boxes, "is_labeled": is_labeled, "idx": idx}

    if "image_weak" in batch[0]:
        result["image_weak"] = torch.stack([s["image_weak"] for s in batch], dim=0)
    if "image_strong" in batch[0]:
        result["image_strong"] = torch.stack([s["image_strong"] for s in batch], dim=0)

    return result
