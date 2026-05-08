import logging

from .yolodata import build_yolo_loader, build_yolo_semi_loader

logger = logging.getLogger("global")


def get_loader(cfg, seed=0):
	cfg_dataset = cfg.get("dataset", {})
	dataset_type = cfg_dataset.get("type", "yolo")

	if dataset_type in {"yolo_semi"}:
		train_loader_sup, train_loader_unsup, val_loader = build_yolo_semi_loader("train", cfg, seed=seed)
		logger.info("Get loader Done...")
		return train_loader_sup, train_loader_unsup, val_loader

	if dataset_type in {"yolo", "yolo_sup"}:
		train_loader_sup = build_yolo_loader("train", cfg, seed=seed)
		val_loader = build_yolo_loader("val", cfg, seed=seed)
		logger.info("Get loader Done...")
		return train_loader_sup, val_loader

	raise NotImplementedError(
		f"dataset type {dataset_type} is not supported. Use 'yolo' or 'yolo_semi'."
	)
