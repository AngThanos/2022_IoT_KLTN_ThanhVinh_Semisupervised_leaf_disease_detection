"""
Data augmentation transforms for YOLO detection training.
Optimized for YOLO with diverse AutoAugment operations.
"""
import random
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import torchvision.transforms.functional as TF


class Compose:
    """Compose multiple transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label=None):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class Resize:
    """Resize image to target size, preserving normalized bounding boxes."""
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
        # Normalized boxes stay the same after resize
        return image, label


class RandomFlip:
    """Random horizontal flip with box adjustment."""
    def __init__(self, prob=0.5, flag_hflip=True):
        self.prob = prob
        self.flag_hflip = flag_hflip

    def __call__(self, image, label=None):
        if random.random() < self.prob and self.flag_hflip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if label is not None and len(label) > 0:
                # Flip x_center: x_center = 1 - x_center
                label = label.copy()
                label[:, 1] = 1.0 - label[:, 1]
        return image, label


class Crop:
    """Crop image and adjust normalized bounding boxes."""
    def __init__(self, crop_size, crop_type="center", mean=None, ignore_value=255):
        self.crop_size = crop_size if isinstance(crop_size, (list, tuple)) else [crop_size, crop_size]
        self.crop_type = crop_type
        self.mean = mean or [0.485, 0.456, 0.406]
        self.ignore_value = ignore_value

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

        # Crop or pad
        if w >= crop_w and h >= crop_h:
            image = image.crop((x, y, x + crop_w, y + crop_h))
        else:
            new_image = Image.new('RGB', (crop_w, crop_h),
                                  tuple(int(m * 255) for m in self.mean))
            new_image.paste(image, (0, 0))
            image = new_image
            x, y = 0, 0

        # Adjust bounding box labels
        if label is not None and len(label) > 0:
            label = label.copy()
            # label format: [class, cx_norm, cy_norm, w_norm, h_norm]
            cx_px = label[:, 1] * w
            cy_px = label[:, 2] * h
            bw_px = label[:, 3] * w
            bh_px = label[:, 4] * h

            x1 = cx_px - bw_px / 2.0
            y1 = cy_px - bh_px / 2.0
            x2 = cx_px + bw_px / 2.0
            y2 = cy_px + bh_px / 2.0

            # Move boxes into crop coordinate frame
            x1 = x1 - x
            y1 = y1 - y
            x2 = x2 - x
            y2 = y2 - y

            # Clip boxes to cropped bounds
            x1 = np.clip(x1, 0.0, float(crop_w))
            y1 = np.clip(y1, 0.0, float(crop_h))
            x2 = np.clip(x2, 0.0, float(crop_w))
            y2 = np.clip(y2, 0.0, float(crop_h))

            new_w = x2 - x1
            new_h = y2 - y1
            valid = (new_w > 1.0) & (new_h > 1.0)

            if np.any(valid):
                cls = label[valid, 0:1]
                x1 = x1[valid]
                y1 = y1[valid]
                x2 = x2[valid]
                y2 = y2[valid]
                new_w = new_w[valid]
                new_h = new_h[valid]

                new_cx = (x1 + x2) / 2.0
                new_cy = (y1 + y2) / 2.0

                out = np.zeros((len(new_cx), 5), dtype=np.float32)
                out[:, 0:1] = cls
                out[:, 1] = new_cx / float(crop_w)
                out[:, 2] = new_cy / float(crop_h)
                out[:, 3] = new_w / float(crop_w)
                out[:, 4] = new_h / float(crop_h)
                out[:, 1:] = np.clip(out[:, 1:], 0.0, 1.0)
                label = out
            else:
                label = np.zeros((0, 5), dtype=np.float32)

        return image, label


class ToTensorAndNormalize:
    """Convert PIL Image to tensor [0, 1] for YOLO models."""
    def __init__(self, mean=None, std=None):
        # YOLO models expect [0, 1] range; ignore ImageNet mean/std for detection
        pass

    def __call__(self, image, label=None):
        # Convert to tensor: PIL [0, 255] -> Tensor [0, 1]
        image = TF.to_tensor(image)
        
        # Convert label to tensor
        if label is not None:
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label).float()
            elif not isinstance(label, torch.Tensor):
                label = torch.tensor(label, dtype=torch.float32)
        else:
            label = torch.zeros((0, 5), dtype=torch.float32)
        
        return image, label

# ============================================================================
# Color & Light Augmentations
# ============================================================================

class ColorJitter:
    """Random color jitter."""
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, label=None):
        image = TF.adjust_brightness(image, 1 + random.uniform(-self.brightness, self.brightness))
        image = TF.adjust_contrast(image, 1 + random.uniform(-self.contrast, self.contrast))
        image = TF.adjust_saturation(image, 1 + random.uniform(-self.saturation, self.saturation))
        image = TF.adjust_hue(image, random.uniform(-self.hue, self.hue))
        return image, label


class RandomBrightness:
    """Random brightness adjustment."""
    def __init__(self, strength=0.35):
        self.strength = strength

    def __call__(self, image, label=None):
        factor = 1.0 + random.uniform(-self.strength, self.strength)
        image = TF.adjust_brightness(image, factor)
        return image, label


class RandomContrast:
    """Random contrast adjustment."""
    def __init__(self, strength=0.35):
        self.strength = strength

    def __call__(self, image, label=None):
        factor = 1.0 + random.uniform(-self.strength, self.strength)
        image = TF.adjust_contrast(image, factor)
        return image, label


class RandomHue:
    """Random hue adjustment."""
    def __init__(self, strength=0.08):
        self.strength = strength

    def __call__(self, image, label=None):
        hue_factor = random.uniform(-self.strength, self.strength)
        image = TF.adjust_hue(image, hue_factor)
        return image, label


class RandomSaturation:
    """Random saturation adjustment."""
    def __init__(self, strength=0.4):
        self.strength = strength

    def __call__(self, image, label=None):
        factor = 1.0 + random.uniform(-self.strength, self.strength)
        image = TF.adjust_saturation(image, factor)
        return image, label


class RandomSharpness:
    """Random sharpness adjustment."""
    def __init__(self, strength=0.8):
        self.strength = strength

    def __call__(self, image, label=None):
        sharpness_factor = max(0.1, 1.0 + random.uniform(-self.strength, self.strength))
        image = TF.adjust_sharpness(image, sharpness_factor)
        return image, label


class GaussianBlur:
    """Apply Gaussian blur."""
    def __init__(self, radius_range=(0.1, 2.0), prob=1.0):
        self.radius_range = radius_range
        self.prob = prob

    def __call__(self, image, label=None):
        if random.random() < self.prob:
            radius = random.uniform(*self.radius_range)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image, label


class RandomGrayscale:
    """Convert to grayscale with probability p."""
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, image, label=None):
        if random.random() < self.p:
            image = ImageOps.grayscale(image)
            image = image.convert('RGB')
        return image, label


# ============================================================================
# Advanced AutoAugment-style Operations
# ============================================================================

class Autocontrast:
    """Auto contrast enhancement."""
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, image, label=None):
        if random.random() < self.prob:
            image = ImageOps.autocontrast(image)
        return image, label


class Equalize:
    """Histogram equalization."""
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, image, label=None):
        if random.random() < self.prob:
            image = ImageOps.equalize(image)
        return image, label


class Invert:
    """Invert image colors."""
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, image, label=None):
        if random.random() < self.prob:
            image = ImageOps.invert(image.convert('RGB'))
        return image, label


class Posterize:
    """Reduce number of bits for each color channel."""
    def __init__(self, bits_range=(4, 8), prob=1.0):
        self.bits_range = bits_range
        self.prob = prob

    def __call__(self, image, label=None):
        if random.random() < self.prob:
            bits = random.randint(self.bits_range[0], self.bits_range[1])
            image = ImageOps.posterize(image, bits)
        return image, label


class Solarize:
    """Solarize image (invert above threshold)."""
    def __init__(self, threshold_range=(1, 256), prob=1.0):
        self.threshold_range = threshold_range
        self.prob = prob

    def __call__(self, image, label=None):
        if random.random() < self.prob:
            threshold = random.randint(self.threshold_range[0], self.threshold_range[1])
            image = ImageOps.solarize(image, threshold)
        return image, label





# ============================================================================
# Strong Augmentation for Semi-Supervised Learning
# ============================================================================

class strong_img_aug:
    """
    Strong augmentation for semi-supervised learning.
    Randomly applies multiple augmentations to create diverse perturbations.
    If a hardness value is provided, easier samples get lighter perturbations
    and harder samples get stronger perturbations.
    """
    
    def __init__(self, num_augs=2, aug_prob=None, min_augs=1, max_augs=5):
        """
        Args:
            num_augs: Number of augmentations to apply per image
            aug_prob: Dict of {augmentation_name: probability} for selective sampling
        """
        self.num_augs = num_augs
        self.min_augs = min_augs
        self.max_augs = max_augs
        self.aug_prob = aug_prob or {}
        
        # Lightweight pool for easy samples.
        self.weak_pool = [
            ("ColorJitter", ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05)),
            ("RandomBrightness", RandomBrightness(strength=0.12)),
            ("RandomContrast", RandomContrast(strength=0.12)),
            ("RandomHue", RandomHue(strength=0.04)),
            ("RandomSaturation", RandomSaturation(strength=0.15)),
            ("RandomSharpness", RandomSharpness(strength=0.2)),
            ("GaussianBlur", GaussianBlur(radius_range=(0.1, 0.8), prob=0.7)),
            ("RandomGrayscale", RandomGrayscale(p=0.05)),
        ]

        # Balanced pool for normal samples.
        self.mid_pool = [
            ("ColorJitter", ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08)),
            ("RandomBrightness", RandomBrightness(strength=0.25)),
            ("RandomContrast", RandomContrast(strength=0.25)),
            ("RandomHue", RandomHue(strength=0.06)),
            ("RandomSaturation", RandomSaturation(strength=0.25)),
            ("RandomSharpness", RandomSharpness(strength=0.45)),
            ("GaussianBlur", GaussianBlur(radius_range=(0.1, 1.4), prob=0.9)),
            ("RandomGrayscale", RandomGrayscale(p=0.1)),
            ("Autocontrast", Autocontrast(prob=1.0)),
            ("Equalize", Equalize(prob=1.0)),
        ]

        # Strong pool for hard samples.
        self.strong_pool = [
            # Basic color/light augmentations
            ("ColorJitter", ColorJitter(brightness=0.45, contrast=0.45, saturation=0.45, hue=0.1)),
            ("RandomBrightness", RandomBrightness(strength=0.4)),
            ("RandomContrast", RandomContrast(strength=0.4)),
            ("RandomHue", RandomHue(strength=0.1)),
            ("RandomSaturation", RandomSaturation(strength=0.45)),
            ("RandomSharpness", RandomSharpness(strength=0.8)),
            ("GaussianBlur", GaussianBlur(radius_range=(0.1, 2.0), prob=1.0)),
            ("RandomGrayscale", RandomGrayscale(p=0.15)),

            # Advanced AutoAugment operations
            ("Autocontrast", Autocontrast(prob=1.0)),
            ("Equalize", Equalize(prob=1.0)),
            ("Invert", Invert(prob=1.0)),
            ("Posterize", Posterize(bits_range=(4, 8))),
            ("Solarize", Solarize(threshold_range=(50, 200))),
        ]
        
        self.aug_pool = self.strong_pool
        self.names = [name for name, _ in self.aug_pool]
        self.transforms = [t for _, t in self.aug_pool]

    def _pick_pool(self, hardness=None):
        if hardness is None:
            return self.aug_pool, self.num_augs

        hardness = float(np.clip(hardness, 0.0, 1.0))
        if hardness < 0.33:
            pool = self.weak_pool
            n = self.min_augs
        elif hardness < 0.66:
            pool = self.mid_pool
            n = self.num_augs
        else:
            pool = self.strong_pool
            n = min(self.max_augs, max(self.num_augs, int(round(self.num_augs + hardness * 2))))
        return pool, max(1, min(n, len(pool)))

    def __call__(self, image, label=None, hardness=None):
        """
        Apply random augmentations to image.
        Args:
            image: PIL Image
            label: Bounding boxes (not modified by strong aug)
        Returns:
            Augmented image, unchanged label
        """
        pool, n_augs = self._pick_pool(hardness=hardness)
        selected = random.sample(pool, k=min(n_augs, len(pool)))

        for _, aug_transform in selected:
            image, _ = aug_transform(image, None)
        
        return image, label


# ============================================================================
# Utility Functions for augmentation pipelines
# ============================================================================

def build_augmentation_pipeline(config=None):
    """
    Build augmentation pipeline from config dict.
    
    Example config:
    {
        'resize': {'base_size': 640},
        'random_flip': {'prob': 0.5},
        'crop': {'crop_size': 640, 'crop_type': 'rand'},
        'to_tensor': {}
    }
    """
    config = config or {}
    transforms = []
    
    if config.get('resize'):
        cfg = config['resize']
        transforms.append(Resize(cfg.get('base_size', 640)))
    
    if config.get('random_flip'):
        cfg = config['random_flip']
        transforms.append(RandomFlip(prob=cfg.get('prob', 0.5)))
    
    if config.get('crop'):
        cfg = config['crop']
        transforms.append(Crop(cfg.get('crop_size', 640), crop_type=cfg.get('crop_type', 'center')))
    
    if config.get('to_tensor', True):
        transforms.append(ToTensorAndNormalize())
    
    return Compose(transforms) if transforms else None
