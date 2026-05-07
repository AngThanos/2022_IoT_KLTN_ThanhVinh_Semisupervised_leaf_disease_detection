import logging
from .yolodata import build_semi_yololoader, build_yololoader

logger = logging.getLogger("global")


def get_loader(cfg, seed=0):
    cfg_dataset = cfg["dataset"]

    if cfg_dataset["type"] == "yolo_semi":
        train_loader_sup, train_loader_unsup = build_semi_yololoader("train", cfg, seed=seed)
        val_loader = build_yololoader("val", cfg)
        logger.info("Get loader Done...")
        return train_loader_sup, train_loader_unsup, val_loader
    
    elif cfg_dataset["type"] == "yolo":
        train_loader_sup = build_yololoader("train", cfg, seed=seed)
        val_loader = build_yololoader("val", cfg)
        logger.info("Get loader Done...")
        return train_loader_sup, val_loader

    else:
        raise NotImplementedError(f"dataset type {cfg_dataset['type']} is not supported. Use 'yolo' or 'yolo_semi'.")
