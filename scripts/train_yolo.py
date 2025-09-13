import os
from dataclasses import dataclass
import tyro

@dataclass
class YoloTrainCfg:
    data_yaml: str = 'Rust and Scrach.v1i.yolov8/data.yaml'
    model: str = 'yolov8n.pt'
    imgsz: int = 640
    epochs: int = 50
    batch: int = 8
    device: int | str = 0  # or 'cpu'
    out_dir: str = 'yolov8_runs'


def main(cfg: YoloTrainCfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    from ultralytics import YOLO  # type: ignore
    model = YOLO(cfg.model)
    model.train(
        data=cfg.data_yaml,
        imgsz=cfg.imgsz,
        epochs=cfg.epochs,
        batch=cfg.batch,
        device=cfg.device,
        project=cfg.out_dir,
        name='rust_scratch',
        pretrained=True,
        exist_ok=True,
    )


if __name__ == '__main__':
    cfg = tyro.cli(YoloTrainCfg)
    main(cfg)
