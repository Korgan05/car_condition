import os
from dataclasses import dataclass
import tyro

@dataclass
class YoloTrainCfg:
    data_yaml: str = 'Car Scratch and Dent.v1i.yolov8/data.yaml'
    model: str = 'yolov8n.pt'
    imgsz: int = 640
    epochs: int = 50
    batch: int = 8
    device: int | str = 0  # or 'cpu'
    out_dir: str = 'yolov8_runs'
    # augmentation & training knobs
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 10.0
    translate: float = 0.10
    scale: float = 0.50
    shear: float = 2.0
    perspective: float = 0.0
    mosaic: float = 1.0
    mixup: float = 0.10
    lr0: float | None = None  # let optimizer=auto choose when None
    weight_decay: float = 0.0005
    patience: int = 100


def main(cfg: YoloTrainCfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    from ultralytics import YOLO  # type: ignore
    model = YOLO(cfg.model)
    train_kwargs = dict(
        data=cfg.data_yaml,
        imgsz=cfg.imgsz,
        epochs=cfg.epochs,
        batch=cfg.batch,
        device=cfg.device,
        project=cfg.out_dir,
        name='car_scratch_dent',
        pretrained=True,
        exist_ok=True,
        hsv_h=cfg.hsv_h,
        hsv_s=cfg.hsv_s,
        hsv_v=cfg.hsv_v,
        degrees=cfg.degrees,
        translate=cfg.translate,
        scale=cfg.scale,
        shear=cfg.shear,
        perspective=cfg.perspective,
        mosaic=cfg.mosaic,
        mixup=cfg.mixup,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
    )
    if cfg.lr0 is not None:
        train_kwargs['lr0'] = cfg.lr0
    model.train(**train_kwargs)


if __name__ == '__main__':
    cfg = tyro.cli(YoloTrainCfg)
    main(cfg)
