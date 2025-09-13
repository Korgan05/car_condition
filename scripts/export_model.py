import os
from dataclasses import dataclass

import torch
import tyro

from src.model import MultiHeadResNet


@dataclass
class ExportConfig:
    ckpt: str = 'checkpoints/resnet18_multitask.pt'
    out_dir: str = 'checkpoints/export'
    image_size: int = 224
    opset: int = 12
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone: str = 'auto'  # auto|resnet18|resnet34|resnet50


def main(cfg: ExportConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Load model
    state = torch.load(cfg.ckpt, map_location=cfg.device)

    def try_load(backbone_name: str):
        m = MultiHeadResNet(pretrained=False, backbone=backbone_name)
        m.load_state_dict(state['model'])
        return m

    model = None
    if cfg.backbone != 'auto':
        model = try_load(cfg.backbone)
    else:
        last_err = None
        for bb in ['resnet18', 'resnet34', 'resnet50']:
            try:
                model = try_load(bb)
                print(f"Loaded checkpoint with backbone={bb}")
                break
            except Exception as e:
                last_err = e
        if model is None:
            raise last_err

    model = model.to(cfg.device)
    model.eval()

    dummy = torch.randn(1, 3, cfg.image_size, cfg.image_size, device=cfg.device)

    # TorchScript export
    traced = torch.jit.trace(model, dummy)
    ts_path = os.path.join(cfg.out_dir, 'model_ts.pt')
    traced.save(ts_path)

    # Minimal TorchScript check: run forward
    with torch.no_grad():
        o1, o2 = traced(dummy)
        assert o1.shape == (1, 1) and o2.shape == (1, 1), 'Unexpected TS output shapes'

    # ONNX export (two-head output)
    onnx_path = os.path.join(cfg.out_dir, 'model.onnx')
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=['input'],
        output_names=['clean_logit', 'damaged_logit'],
        dynamic_axes={'input': {0: 'batch'}},
        opset_version=cfg.opset,
    )

    # Optional ONNX runtime check if available
    try:
        import onnxruntime as ort  # type: ignore
        import numpy as np

        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        inp = dummy.detach().cpu().numpy().astype('float32')
        outs = sess.run(None, {'input': inp})
        assert len(outs) == 2 and outs[0].shape == (1, 1) and outs[1].shape == (1, 1)
    except Exception as e:
        print('[WARN] ONNX runtime check skipped or failed:', e)

    print('Exported:')
    print('  TorchScript:', ts_path)
    print('  ONNX      :', onnx_path)


if __name__ == '__main__':
    cfg = tyro.cli(ExportConfig)
    main(cfg)
