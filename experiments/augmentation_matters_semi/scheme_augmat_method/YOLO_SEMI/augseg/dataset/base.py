import logging
import os
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset


# Allow loading slightly truncated images to avoid worker crashes on minor JPEG corruption.
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseDataset(Dataset):
    """Base dataset class for YOLO detection."""
    
    def __init__(self, d_list, **kwargs):
        self.parse_input_list(d_list, **kwargs)

    def parse_input_list(self, d_list, max_sample=-1, start_idx=-1, end_idx=-1):
        logger = logging.getLogger("global")
        assert isinstance(d_list, str)
        
        # Determine if validation or training data
        if "val" in d_list:
            self.list_sample = [
                [
                    f"images/{line.strip()}.JPG",
                    f"labels/{line.strip()}.txt",
                ]
                for line in open(d_list, "r")
            ]
        elif "unlabeled" in d_list:
            self.list_sample = [
                [
                    f"unlabeled/images/{line.strip()}.JPG",
                    f"unlabeled/labels/{line.strip()}.txt",
                ]
                for line in open(d_list, "r")
            ]
        else:
            self.list_sample = [
                [
                    f"labeled/images/{line.strip()}.JPG",
                    f"labeled/labels/{line.strip()}.txt",
                ]
                for line in open(d_list, "r")
            ]

        if max_sample > 0:
            self.list_sample = self.list_sample[:max_sample]
        if start_idx >= 0 and end_idx >= 0:
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        logger.info(f"# samples: {self.num_sample}")

    def img_loader(self, path, mode="RGB"):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert(mode)

    def label_loader(self, path):
        """Load YOLO format label: class x_center y_center width height"""
        if path is None or not os.path.exists(path):
            return np.zeros((0, 5), dtype=np.float32)
        
        try:
            with open(path, 'r') as f:
                lines = f.read().strip().split('\n')
            
            if not lines or lines[0] == '':
                return np.zeros((0, 5), dtype=np.float32)
            
            labels = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels.append([float(x) for x in parts[:5]])
            
            return np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)
            
        except Exception as e:
            print(f"Error loading label {path}: {e}")
            return np.zeros((0, 5), dtype=np.float32)

    def __len__(self):
        return self.num_sample
