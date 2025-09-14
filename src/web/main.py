import io
import os
from dataclasses import dataclass
from typing import Optional

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
import tyro
import numpy as np
import cv2

from ..model import MultiHeadResNet


def build_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def make_app(
    ckpt: str,
    image_size: int = 224,
    device: str = 'cpu',
    backbone: str = 'resnet18',
    yolo_weights: Optional[str] = None,
    thr_clean_override: Optional[float] = None,
    thr_damaged_override: Optional[float] = None,
    invert_clean: bool = False,
    invert_damaged: bool = False,
):
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    # auto-detect backbone if requested or if mismatch occurs
    def try_load(bb: str):
        m = MultiHeadResNet(pretrained=False, backbone=bb)
        m.load_state_dict(state['model'])
        return m
    model = None
    bbs = [backbone] if backbone != 'auto' else ['resnet18', 'resnet34', 'resnet50']
    last_err = None
    for bb in bbs:
        try:
            model = try_load(bb)
            backbone = bb
            break
        except Exception as e:
            last_err = e
            continue
    if model is None:
        raise last_err
    model.to(device).eval()

    t = build_transform(image_size)

    app = FastAPI(title='Car Condition Web')

    # CORS (локально и для деплоя)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    app.mount('/static', StaticFiles(directory=static_dir), name='static')

    @app.get('/', response_class=HTMLResponse)
    def index():
        with open(os.path.join(static_dir, 'index.html'), 'r', encoding='utf-8') as f:
            return f.read()

    @app.get('/health')
    def health():
        return {'status': 'ok', 'image_size': image_size, 'device': device}

    # Optional YOLO detector
    yolo_model = None
    yolo_classes = None
    if yolo_weights and os.path.exists(yolo_weights):
        try:
            from ultralytics import YOLO  # type: ignore
            yolo_model = YOLO(yolo_weights)
            # names: dict[int,str]
            yolo_classes = [yolo_model.model.names[i] for i in range(len(yolo_model.model.names))]
        except Exception as e:
            print('[WARN] YOLO load failed:', e)
            yolo_model = None
            yolo_classes = None

    def _friendly_class(name: Optional[str]) -> Optional[str]:
        if name is None:
            return None
        name = name.lower()
        mapping = {
            'scracth': 'царапина',
            'scratch': 'царапина',
            'dunt': 'вмятина',
            'dent': 'вмятина',
            'rust': 'ржавчина',
            'car': 'авто',
        }
        return mapping.get(name, name)

    @app.get('/api/metadata')
    def metadata():
        return {
            'backbone': backbone,
            'thresholds': {'clean': thr_c, 'damaged': thr_d},
            'yolo_loaded': bool(yolo_model is not None),
            'yolo_classes': yolo_classes,
        }

    # Allow trailing slashes for endpoints (avoids 404 when client posts to /api/.../)
    @app.get('/api/metadata/', include_in_schema=False)
    def metadata_slash():
        return metadata()

    thr_c = state.get('thresholds', {}).get('clean', 0.5)
    thr_d = state.get('thresholds', {}).get('damaged', 0.5)
    if isinstance(thr_clean_override, (int, float)):
        thr_c = float(thr_clean_override)
    if isinstance(thr_damaged_override, (int, float)):
        thr_d = float(thr_damaged_override)

    # -------- Grad-CAM helpers --------
    def _get_last_conv():
        last = model.backbone.layer4[-1]
        if hasattr(last, 'conv2'):
            return last.conv2
        if hasattr(last, 'conv3'):
            return last.conv3
        # fallback: try to find last Conv2d in layer4
        for m in reversed(list(model.backbone.layer4.modules())):
            if isinstance(m, torch.nn.Conv2d):
                return m
        raise RuntimeError('No conv layer found for Grad-CAM')

    def _compute_cam(img_t: torch.Tensor, head: str):
        feats = None
        grads = None

        def fwd_hook(module, inp, out):
            nonlocal feats
            feats = out.detach()

        def bwd_hook(module, gin, gout):
            nonlocal grads
            grads = gout[0].detach()

        conv = _get_last_conv()
        h1 = conv.register_forward_hook(fwd_hook)
        h2 = conv.register_full_backward_hook(bwd_hook)

        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            lc, ld = model(img_t)
            # support head in {'clean','dirty','damaged'}
            if head == 'damaged':
                logit = ld
            else:
                # for 'clean' or 'dirty' we backprop through 'clean' logit
                logit = lc
            logit.backward(torch.ones_like(logit))

        h1.remove(); h2.remove()
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        cam = (weights * feats).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)
        return cam[0].cpu().numpy()

    def _overlay_cam(img_np: np.ndarray, cam: np.ndarray):
        h, w = img_np.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        heat = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        out = (0.45 * heat + 0.55 * img_np).astype(np.uint8)
        return out

    @app.post('/api/predict')
    async def predict(file: UploadFile = File(...)):
        if file.content_type not in {'image/jpeg', 'image/png', 'image/bmp'}:
            raise HTTPException(status_code=400, detail='Неподдерживаемый тип файла (разрешены JPEG/PNG/BMP)')
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert('RGB')
        x = t(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pc, pd = model.predict_proba(x)
            pc = float(pc.item())
            pd = float(pd.item())
        # apply optional inversion of semantics
        pc_eff = 1.0 - pc if invert_clean else pc
        pd_eff = 1.0 - pd if invert_damaged else pd
        return JSONResponse({
            'clean_prob': pc_eff,
            'dirty_prob': 1 - pc_eff,
            'damaged_prob': pd_eff,
            'intact_prob': 1 - pd_eff,
            'pred_clean': 'clean' if pc_eff >= thr_c else 'dirty',
            'pred_damage': 'damaged' if pd_eff >= thr_d else 'intact',
            # localized labels
            'pred_clean_ru': 'чистый' if pc_eff >= thr_c else 'грязный',
            'pred_damage_ru': 'битый' if pd_eff >= thr_d else 'целый',
        })

    @app.post('/api/predict/', include_in_schema=False)
    async def predict_slash(file: UploadFile = File(...)):
        return await predict(file)

    @app.post('/api/batch_predict')
    async def batch_predict(files: list[UploadFile] = File(...)):
        results = []
        for file in files:
            try:
                if file.content_type not in {'image/jpeg', 'image/png', 'image/bmp'}:
                    raise ValueError('Неподдерживаемый тип файла (разрешены JPEG/PNG/BMP)')
                data = await file.read()
                img = Image.open(io.BytesIO(data)).convert('RGB')
                x = t(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    pc, pd = model.predict_proba(x)
                    pc = float(pc.item())
                    pd = float(pd.item())
                pc_eff = 1.0 - pc if invert_clean else pc
                pd_eff = 1.0 - pd if invert_damaged else pd
                results.append({
                    'filename': file.filename,
                    'clean_prob': pc_eff,
                    'dirty_prob': 1 - pc_eff,
                    'damaged_prob': pd_eff,
                    'intact_prob': 1 - pd_eff,
                    'pred_clean': 'clean' if pc_eff >= thr_c else 'dirty',
                    'pred_damage': 'damaged' if pd_eff >= thr_d else 'intact',
                    # localized labels
                    'pred_clean_ru': 'чистый' if pc_eff >= thr_c else 'грязный',
                    'pred_damage_ru': 'битый' if pd_eff >= thr_d else 'целый',
                })
            except Exception as e:
                results.append({'filename': getattr(file, 'filename', None), 'error': str(e)})
        return JSONResponse({'results': results})

    @app.post('/api/batch_predict/', include_in_schema=False)
    async def batch_predict_slash(files: list[UploadFile] = File(...)):
        return await batch_predict(files)

    @app.post('/api/heatmap')
    async def heatmap(head: str = 'damaged', file: UploadFile = File(...)):
        if file.content_type not in {'image/jpeg', 'image/png', 'image/bmp'}:
            raise HTTPException(status_code=400, detail='Неподдерживаемый тип файла (разрешены JPEG/PNG/BMP)')
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert('RGB')
        img_np = np.array(img)
        x = t(img).unsqueeze(0).to(device)
        # if head == 'dirty' we invert CAM of 'clean' to highlight грязь
        if head == 'dirty':
            cam = _compute_cam(x, 'clean')
            cam = 1.0 - cam
        else:
            cam = _compute_cam(x, head)
        vis = _overlay_cam(img_np, cam)
        buf = io.BytesIO()
        Image.fromarray(vis).save(buf, format='PNG')
        buf.seek(0)
        from fastapi.responses import Response
        return Response(content=buf.read(), media_type='image/png')
    @app.post('/api/heatmap/', include_in_schema=False)
    async def heatmap_slash(head: str = 'damaged', file: UploadFile = File(...)):
        return await heatmap(head=head, file=file)

    @app.post('/api/detect')
    async def detect(file: UploadFile = File(...), conf: float = 0.10, iou: float = 0.45):
        if yolo_model is None:
            raise HTTPException(status_code=503, detail='Детектор не загружен. Передайте путь к YOLO весам при запуске.')
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert('RGB')
        # Run inference
        results = yolo_model.predict(img, conf=conf, iou=iou, verbose=False)
        dets = []
        if results:
            r = results[0]
            boxes = r.boxes
            for i in range(len(boxes)):
                b = boxes[i]
                xyxy = b.xyxy[0].tolist()
                cls_id = int(b.cls.item())
                score = float(b.conf.item())
                raw_name = r.names.get(cls_id) if hasattr(r, 'names') and isinstance(r.names, dict) else None
                dets.append({'box': [float(v) for v in xyxy], 'class_id': cls_id, 'class': _friendly_class(raw_name), 'score': score})
        # retry with lower conf if empty
        if not dets and (conf is None or conf > 0.06):
            results = yolo_model.predict(img, conf=0.05, iou=iou, verbose=False)
            if results:
                r = results[0]
                boxes = r.boxes
                for i in range(len(boxes)):
                    b = boxes[i]
                    xyxy = b.xyxy[0].tolist()
                    cls_id = int(b.cls.item())
                    score = float(b.conf.item())
                    raw_name = r.names.get(cls_id) if hasattr(r, 'names') and isinstance(r.names, dict) else None
                    dets.append({'box': [float(v) for v in xyxy], 'class_id': cls_id, 'class': _friendly_class(raw_name), 'score': score})
        # also return friendly classes
        classes_out = None
        try:
            classes_out = [_friendly_class(yolo_model.model.names[i]) for i in range(len(yolo_model.model.names))]
        except Exception:
            classes_out = yolo_classes
        return JSONResponse({'detections': dets, 'classes': classes_out})
    @app.post('/api/detect/', include_in_schema=False)
    async def detect_slash(file: UploadFile = File(...), conf: float = 0.25, iou: float = 0.45):
        return await detect(file=file, conf=conf, iou=iou)

    @app.post('/api/detect_image')
    async def detect_image(file: UploadFile = File(...), conf: float = 0.10, iou: float = 0.45):
        if yolo_model is None:
            raise HTTPException(status_code=503, detail='Детектор не загружен. Передайте путь к YOLO весам при запуске.')
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert('RGB')
        results = yolo_model.predict(img, conf=conf, iou=iou, verbose=False)
        if not results:
            raise HTTPException(status_code=500, detail='YOLO inference failed')
        r = results[0]
        # if no boxes, retry with lower conf
        try:
            if r.boxes is not None and len(r.boxes) == 0 and (conf is None or conf > 0.06):
                results = yolo_model.predict(img, conf=0.05, iou=iou, verbose=False)
                r = results[0]
        except Exception:
            pass
        # replace names to friendly RU labels for plotting
        try:
            if hasattr(r, 'names') and isinstance(r.names, dict):
                r.names = {i: _friendly_class(r.names.get(i)) for i in r.names}
        except Exception:
            pass
        # r.plot() -> ndarray BGR (увеличим толщину линий для наглядности)
        try:
            plot_bgr = r.plot(line_width=4)
        except TypeError:
            plot_bgr = r.plot()
        plot_rgb = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        Image.fromarray(plot_rgb).save(buf, format='PNG')
        buf.seek(0)
        from fastapi.responses import Response
        return Response(content=buf.read(), media_type='image/png')
    @app.post('/api/detect_image/', include_in_schema=False)
    async def detect_image_slash(file: UploadFile = File(...), conf: float = 0.25, iou: float = 0.45):
        return await detect_image(file=file, conf=conf, iou=iou)

    return app


@dataclass
class WebConfig:
    ckpt: str = 'checkpoints/resnet18_multitask.pt'
    host: str = '127.0.0.1'
    port: int = 8000
    image_size: int = 224
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone: str = 'auto'
    yolo_weights: Optional[str] = None
    # overrides & interpretation
    thr_clean: Optional[float] = None
    thr_damaged: Optional[float] = None
    invert_clean: bool = False
    invert_damaged: bool = False


def main(cfg: WebConfig):
    import uvicorn
    app = make_app(
        cfg.ckpt,
        cfg.image_size,
        cfg.device,
        cfg.backbone,
    yolo_weights=cfg.yolo_weights,
        thr_clean_override=cfg.thr_clean,
        thr_damaged_override=cfg.thr_damaged,
        invert_clean=cfg.invert_clean,
        invert_damaged=cfg.invert_damaged,
    )
    uvicorn.run(app, host=cfg.host, port=cfg.port)


if __name__ == '__main__':
    cfg = tyro.cli(WebConfig)
    main(cfg)
