"""
Data augmentation transforms for YOLO detection training.
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
    """Resize image to target size."""
    def __init__(self, base_size, rand_resize=None):
        # Handle both int and list/tuple for base_size
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
    """Random horizontal flip."""
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
    """Crop image to target size."""
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

        # Ensure valid crop region
        x = max(0, min(x, w - crop_w)) if w > crop_w else 0
        y = max(0, min(y, h - crop_h)) if h > crop_h else 0

        # Crop or pad
        if w >= crop_w and h >= crop_h:
            image = image.crop((x, y, x + crop_w, y + crop_h))
        else:
            # Pad if needed (paste original at top-left)
            new_image = Image.new('RGB', (crop_w, crop_h),
                                  tuple(int(m * 255) for m in self.mean))
            new_image.paste(image, (0, 0))
            image = new_image
            x, y = 0, 0  # no offset when padding

        # Adjust bounding box labels for crop/pad with geometric clipping.
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

            # Move boxes into crop coordinate frame.
            x1 = x1 - x
            y1 = y1 - y
            x2 = x2 - x
            y2 = y2 - y

            # Clip boxes to cropped/padded image bounds.
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
        # YOLO models expect [0, 1] range; ignore ImageNet mean/std
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


class GaussianBlur:
    """Apply Gaussian blur."""
    def __init__(self, radius_range=(0.1, 2.0)):
        self.radius_range = radius_range

    def __call__(self, image, label=None):
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


class RandomSharpness:
    """Random sharpness adjustment."""
    def __init__(self, strength=0.8):
        self.strength = strength

    def __call__(self, image, label=None):
        sharpness_factor = max(0.1, 1.0 + random.uniform(-self.strength, self.strength))
        image = TF.adjust_sharpness(image, sharpness_factor)
        return image, label


class strong_img_aug:
    """AugSeg TIBA: random number of ops + uniform continuous strength + WITH replacement.
    
    Paper Section 3.2: "we select a random number of different intensity-based 
    augmentations and a random distortion strength from a continuous space."
    
    Key differences from RandAugment:
    - random.choices (WITH replacement) instead of random.sample
    - Uniform continuous intensity instead of discrete levels
    - Random number of ops [1, k] instead of fixed number
    """
    def __init__(self, num_augs=3, flag_using_random_num=True):
        self.n = num_augs
        self.flag_using_random_num = flag_using_random_num
        # Pool from paper Table 2 (11 ops, remove Invert which is too strong)
        self.augment_list = [
            (self._identity, None),
            (self._autocontrast, None),
            (self._equalize, None),
            (self._blur, [0.1, 2.0]),
            (self._contrast, [0.05, 0.95]),
            (self._brightness, [0.05, 0.95]),
            (self._color, [0.05, 0.95]),
            (self._sharpness, [0.05, 0.95]),
            (self._posterize, [4, 8]),
            (self._solarize, [1, 256]),
            (self._hue, [0, 0.5]),
        ]

    def __call__(self, image, label=None):
        # Paper: "sample a random number of augmentations, bounded by k"
        if self.flag_using_random_num:
            max_num = np.random.randint(1, high=self.n + 1)
        else:
            max_num = self.n
        # Paper: random.choices (WITH replacement)
        ops = random.choices(self.augment_list, k=max_num)
        for op, scales in ops:
            image = op(image, scales)
        # Only modifies image (intensity-only), does NOT change bounding boxes
        return image

    @staticmethod
    def _identity(img, scale):
        return img

    @staticmethod
    def _autocontrast(img, scale):
        return ImageOps.autocontrast(img)

    @staticmethod
    def _equalize(img, scale):
        return ImageOps.equalize(img)

    @staticmethod
    def _blur(img, scale):
        sigma = random.uniform(scale[0], scale[1])
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))

    @staticmethod
    def _contrast(img, scale):
        v = random.uniform(scale[0], scale[1])
        return ImageEnhance.Contrast(img).enhance(v)

    @staticmethod
    def _brightness(img, scale):
        v = random.uniform(scale[0], scale[1])
        return ImageEnhance.Brightness(img).enhance(v)

    @staticmethod
    def _color(img, scale):
        v = random.uniform(scale[0], scale[1])
        return ImageEnhance.Color(img).enhance(v)

    @staticmethod
    def _sharpness(img, scale):
        v = random.uniform(scale[0], scale[1])
        return ImageEnhance.Sharpness(img).enhance(v)

    @staticmethod
    def _posterize(img, scale):
        v = int(random.uniform(scale[0], scale[1]))
        v = max(1, v)
        return ImageOps.posterize(img, v)

    @staticmethod
    def _solarize(img, scale):
        v = int(random.uniform(scale[0], scale[1]))
        v = max(1, v)
        return ImageOps.solarize(img, v)

    @staticmethod
    def _hue(img, scale):
        v = random.uniform(scale[0], scale[1])
        if random.random() < 0.5:
            v = -v
        input_mode = img.mode
        if input_mode in {"L", "1", "I", "F"}:
            return img
        h, s, val = img.convert("HSV").split()
        np_h = np.array(h, dtype=np.uint8)
        np_h = np.clip(np_h.astype(np.int16) + int(v * 255), 0, 255).astype(np.uint8)
        h = Image.fromarray(np_h, "L")
        return Image.merge("HSV", (h, s, val)).convert(input_mode)
