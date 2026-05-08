from __future__ import annotations

from typing import Type


def get_yolo_class() -> Type:
    try:
        from ultralytics import YOLO  # type: ignore

        return YOLO
    except Exception as exc:
        raise RuntimeError(
            "Cannot import 'ultralytics' from the active Python environment. "
            "Install ultralytics in the environment instead of importing from the vendored reference folder."
        ) from exc
