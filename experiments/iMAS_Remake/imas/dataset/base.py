import logging
from pathlib import Path

from PIL import Image, ImageFile
from torch.utils.data import Dataset

from . import augmentations as img_trsform


# Avoid dataloader worker crashes on minor JPEG truncation/corruption.
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseDataset(Dataset):
    def __init__(self, d_list, **kwargs):
        # parse the input list
        self.parse_input_list(d_list, **kwargs)

    def parse_input_list(self, d_list, max_sample=-1, start_idx=-1, end_idx=-1, label_root_name="labels"):
        logger = logging.getLogger("global")
        assert isinstance(d_list, str)
        self.list_sample = []
        with open(d_list, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    image_rel, label_rel = parts[0], parts[1]
                else:
                    image_rel = parts[0]
                    image_rel_path = Path(image_rel)
                    if image_rel_path.suffix:
                        label_rel = str(image_rel_path.with_suffix(".txt"))
                    else:
                        label_rel = str(Path(label_root_name) / f"{image_rel_path.stem}.txt")
                self.list_sample.append([image_rel, label_rel])

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        logger.info("# samples: {}".format(self.num_sample))

    def img_loader(self, path, mode):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert(mode)

    @staticmethod
    def yolo_label_loader(path):
        if path is None or not Path(path).exists():
            import numpy as np

            return np.zeros((0, 5), dtype=np.float32)

        with open(path, "r", encoding="utf-8") as handle:
            rows = []
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                except ValueError:
                    continue
        if not rows:
            import numpy as np

            return np.zeros((0, 5), dtype=np.float32)
        import numpy as np

        return np.asarray(rows, dtype=np.float32)

    def __len__(self):
        return self.num_sample