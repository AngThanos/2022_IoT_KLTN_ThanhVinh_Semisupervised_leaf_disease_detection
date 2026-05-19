"""
Detection augmentation transforms adapted from iMAS_Remake.
Provides geometric augmentations (that modify both image and bounding boxes)
and strong image-only augmentations for semi-supervised detection.
"""
import random
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import torchvision.transforms.functional as TF
import torch


# ============================================================================
# Geometric augmentations (modify both image and normalized YOLO boxes)
# ============================================================================

class Compose:
    """Compose multiple transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label=None):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class Resize:
    """Resize image to target size. Normalized bounding boxes are invariant."""
    def __init__(self, base_size, rand_resize=None):
        if isinstance(base_size, int):
            self.base_size = [base_size, base_size]
        else:
            self.base_size = base_size
        self.rand_resize = rand_resize

    def __call__(self, image, label=None):
        if self.rand_resize:
            scale = random.uniform(self.rand_resize[0], self.rand_resize[1])
            new_size = [int(s * scale) for s in self.base_size]
        else:
            new_size = self.base_size
        image = image.resize((new_size[1], new_size[0]), Image.BILINEAR)
        return image, label


class RandomFlip:
    """Random horizontal flip with box adjustment."""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, label=None):
        if random.random() < self.prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if label is not None and len(label) > 0:
                label = label.copy()
                label[:, 1] = 1.0 - label[:, 1]  # flip cx
        return image, label


class Crop:
    """Crop image and adjust normalized bounding boxes."""
    def __init__(self, crop_size, crop_type="rand", mean=None):
        self.crop_size = crop_size if isinstance(crop_size, (list, tuple)) else [crop_size, crop_size]
        self.crop_type = crop_type
        self.mean = mean or [0.485, 0.456, 0.406]

    def __call__(self, image, label=None):
        w, h = image.size
        crop_h, crop_w = self.crop_size

        if self.crop_type == "center":
            x = (w - crop_w) // 2
            y = (h - crop_h) // 2
        elif self.crop_type == "rand":
            x = random.randint(0, max(0, w - crop_w))
            y = random.randint(0, max(0, h - crop_h))
        else:
            x, y = 0, 0

        x = max(0, min(x, w - crop_w)) if w > crop_w else 0
        y = max(0, min(y, h - crop_h)) if h > crop_h else 0

        if w >= crop_w and h >= crop_h:
            image = image.crop((x, y, x + crop_w, y + crop_h))
        else:
            new_image = Image.new('RGB', (crop_w, crop_h),
                                  tuple(int(m * 255) for m in self.mean))
            new_image.paste(image, (0, 0))
            image = new_image
            x, y = 0, 0

        if label is not None and len(label) > 0:
            label = label.copy()
            cx_px = label[:, 1] * w;  cy_px = label[:, 2] * h
            bw_px = label[:, 3] * w;  bh_px = label[:, 4] * h

            x1 = np.clip(cx_px - bw_px / 2 - x, 0, float(crop_w))
            y1 = np.clip(cy_px - bh_px / 2 - y, 0, float(crop_h))
            x2 = np.clip(cx_px + bw_px / 2 - x, 0, float(crop_w))
            y2 = np.clip(cy_px + bh_px / 2 - y, 0, float(crop_h))

            new_w = x2 - x1;  new_h = y2 - y1
            valid = (new_w > 1.0) & (new_h > 1.0)

            if np.any(valid):
                out = np.zeros((valid.sum(), 5), dtype=np.float32)
                out[:, 0] = label[valid, 0]
                out[:, 1] = ((x1[valid] + x2[valid]) / 2) / float(crop_w)
                out[:, 2] = ((y1[valid] + y2[valid]) / 2) / float(crop_h)
                out[:, 3] = new_w[valid] / float(crop_w)
                out[:, 4] = new_h[valid] / float(crop_h)
                out[:, 1:] = np.clip(out[:, 1:], 0.0, 1.0)
                label = out
            else:
                label = np.zeros((0, 5), dtype=np.float32)
        return image, label


class ToTensorNorm:
    """Convert PIL image to [0, 1] tensor (YOLO standard). Boxes → torch tensor."""
    def __call__(self, image, label=None):
        image = TF.to_tensor(image)  # [0, 255] → [0, 1]
        if label is not None:
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label).float()
        else:
            label = torch.zeros((0, 5), dtype=torch.float32)
        return image, label


# ============================================================================
# Color / light augmentations (image only, do NOT modify boxes)
# ============================================================================

class ColorJitter:
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        self.b = brightness;  self.c = contrast;  self.s = saturation;  self.h = hue

    def __call__(self, image, label=None):
        image = TF.adjust_brightness(image, 1 + random.uniform(-self.b, self.b))
        image = TF.adjust_contrast(image, 1 + random.uniform(-self.c, self.c))
        image = TF.adjust_saturation(image, 1 + random.uniform(-self.s, self.s))
        image = TF.adjust_hue(image, random.uniform(-self.h, self.h))
        return image, label


class RandomBrightness:
    def __init__(self, strength=0.35):
        self.strength = strength
    def __call__(self, image, label=None):
        f = 1.0 + random.uniform(-self.strength, self.strength)
        return TF.adjust_brightness(image, f), label


class RandomContrast:
    def __init__(self, strength=0.35):
        self.strength = strength
    def __call__(self, image, label=None):
        f = 1.0 + random.uniform(-self.strength, self.strength)
        return TF.adjust_contrast(image, f), label


class RandomHue:
    def __init__(self, strength=0.08):
        self.strength = strength
    def __call__(self, image, label=None):
        return TF.adjust_hue(image, random.uniform(-self.strength, self.strength)), label


class RandomSaturation:
    def __init__(self, strength=0.4):
        self.strength = strength
    def __call__(self, image, label=None):
        f = 1.0 + random.uniform(-self.strength, self.strength)
        return TF.adjust_saturation(image, f), label


class RandomSharpness:
    def __init__(self, strength=0.8):
        self.strength = strength
    def __call__(self, image, label=None):
        f = max(0.1, 1.0 + random.uniform(-self.strength, self.strength))
        return TF.adjust_sharpness(image, f), label


class GaussianBlur:
    def __init__(self, radius_range=(0.1, 2.0), prob=1.0):
        self.radius_range = radius_range;  self.prob = prob
    def __call__(self, image, label=None):
        if random.random() < self.prob:
            r = random.uniform(*self.radius_range)
            image = image.filter(ImageFilter.GaussianBlur(radius=r))
        return image, label


class RandomGrayscale:
    def __init__(self, p=0.2):
        self.p = p
    def __call__(self, image, label=None):
        if random.random() < self.p:
            image = ImageOps.grayscale(image).convert('RGB')
        return image, label


class Autocontrast:
    def __init__(self, prob=1.0):
        self.prob = prob
    def __call__(self, image, label=None):
        if random.random() < self.prob:
            image = ImageOps.autocontrast(image)
        return image, label


class Equalize:
    def __init__(self, prob=1.0):
        self.prob = prob
    def __call__(self, image, label=None):
        if random.random() < self.prob:
            image = ImageOps.equalize(image)
        return image, label


class Posterize:
    def __init__(self, bits_range=(4, 8), prob=1.0):
        self.bits_range = bits_range;  self.prob = prob
    def __call__(self, image, label=None):
        if random.random() < self.prob:
            bits = random.randint(*self.bits_range)
            image = ImageOps.posterize(image, bits)
        return image, label


class Solarize:
    def __init__(self, threshold_range=(1, 256), prob=1.0):
        self.threshold_range = threshold_range;  self.prob = prob
    def __call__(self, image, label=None):
        if random.random() < self.prob:
            thr = random.randint(*self.threshold_range)
            image = ImageOps.solarize(image, thr)
        return image, label


# ============================================================================
# Strong Augmentation Pool (adapted from iMAS)
# ============================================================================

class StrongImgAug:
    """
    Randomly sample `num_augs` augmentations from a strong pool
    and apply them sequentially to the PIL image.
    Boxes are NOT modified (image-only transforms).
    """
    def __init__(self, num_augs=2):
        self.num_augs = num_augs
        self.pool = [
            ColorJitter(brightness=0.45, contrast=0.45, saturation=0.45, hue=0.1),
            RandomBrightness(strength=0.4),
            RandomContrast(strength=0.4),
            RandomHue(strength=0.1),
            RandomSaturation(strength=0.45),
            RandomSharpness(strength=0.8),
            GaussianBlur(radius_range=(0.1, 2.0), prob=1.0),
            RandomGrayscale(p=0.15),
            Autocontrast(prob=1.0),
            Equalize(prob=1.0),
            Posterize(bits_range=(4, 8), prob=1.0),
            Solarize(threshold_range=(1, 256), prob=1.0),
        ]

    def __call__(self, image, label=None):
        ops = random.sample(self.pool, min(self.num_augs, len(self.pool)))
        for op in ops:
            image, label = op(image, label)
        return image, label
