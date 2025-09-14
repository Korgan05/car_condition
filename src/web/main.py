import io
import os
import re
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

    # Serve favicon to avoid 404 in logs
    @app.get('/favicon.ico')
    def favicon():
        path = os.path.join(static_dir, 'favicon.svg')
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail='favicon not found')
        from fastapi.responses import FileResponse
        return FileResponse(path, media_type='image/svg+xml')

    @app.get('/health')
    def health():
        return {'status': 'ok', 'image_size': image_size, 'device': device}

    # Optional YOLO detector(s)
    yolo_models: list = []
    yolo_classes_list: list[list[str]] | None = []
    yolo_weights_list: list[str] = []
    yolo_candidates_tried: list[str] = []
    yolo_error: Optional[str] = None
    # Prepare candidate weights: prefer explicit CLI param; fallback to auto-discovery
    def _discover_yolo_candidates() -> list[str]:
        # if explicit weights passed, use ONLY them
        if yolo_weights:
            return [w.strip() for w in re.split(r'[;,]', yolo_weights) if w.strip()]

        # otherwise, auto-discover common locations
        cands: list[str] = []
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        # yolov8_runs/*/weights/{best,last}.pt
        try:
            runs_dir = os.path.join(root, 'yolov8_runs')
            if os.path.isdir(runs_dir):
                for sub in os.listdir(runs_dir):
                    wdir = os.path.join(runs_dir, sub, 'weights')
                    for fn in ('best.pt', 'last.pt'):
                        p = os.path.join(wdir, fn)
                        if os.path.exists(p):
                            cands.append(p)
        except Exception:
            pass
        # repo root yolov8n.pt as a fallback
        fallback_pt = os.path.join(root, 'yolov8n.pt')
        if os.path.exists(fallback_pt):
            cands.append(fallback_pt)
        # de-duplicate preserving order
        out: list[str] = []
        seen = set()
        for p in cands:
            if p not in seen:
                out.append(p); seen.add(p)
        return out

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        print('[WARN] ultralytics import failed:', e)
        YOLO = None  # type: ignore
        yolo_error = f'ultralytics import failed: {e}'

    weights_candidates = _discover_yolo_candidates()
    for w in weights_candidates:
        yolo_candidates_tried.append(w)
        if not os.path.exists(w):
            print(f"[WARN] YOLO weights not found: {w}")
            continue
        if YOLO is None:  # type: ignore
            break
        try:
            m = YOLO(w)  # type: ignore
            yolo_models.append(m)
            yolo_weights_list.append(w)
            try:
                cls_names = [m.model.names[i] for i in range(len(m.model.names))]
            except Exception:
                cls_names = []
            if yolo_classes_list is not None:
                yolo_classes_list.append(cls_names)
        except Exception as e:
            print('[WARN] YOLO load failed for', w, e)
            yolo_error = f'load failed for {os.path.basename(w)}: {e}'
    if not yolo_models:
        yolo_classes_list = None

    def _friendly_class(name: Optional[str]) -> Optional[str]:
        if name is None:
            return None
        name = name.lower()
        mapping = {
            'scracth': 'царапина',
            'scratch': 'царапина',
            'scratches': 'царапина',
            'dunt': 'вмятина',
            'dent': 'вмятина',
            'dirt': 'грязь',
            'rust': 'ржавчина',
            'rusty': 'ржавчина',
            'corrosion': 'ржавчина',
            'corroded': 'ржавчина',
            'oxidation': 'ржавчина',
            'oxide': 'ржавчина',
            'stain': 'ржавчина',
            'stains': 'ржавчина',
            'mud': 'грязь',
            'dust': 'грязь',
            'dirty': 'грязь',
            'car': 'авто',
        }
        return mapping.get(name, name)

    @app.get('/api/metadata')
    def metadata():
        # friendly classes union
        classes_out = None
        if yolo_classes_list:
            try:
                s = set()
                for lst in yolo_classes_list:
                    for n in lst:
                        fn = _friendly_class(n)
                        if fn:
                            s.add(fn)
                classes_out = sorted(s)
            except Exception:
                classes_out = None
        # prepare weights label (string for UI)
        yolo_weights_label = None
        if yolo_weights_list:
            if len(yolo_weights_list) == 1:
                yolo_weights_label = os.path.basename(yolo_weights_list[0])
            else:
                yolo_weights_label = ', '.join(os.path.basename(w) for w in yolo_weights_list)
        meta = {
            'backbone': backbone,
            'thresholds': {'clean': thr_c, 'damaged': thr_d},
            'yolo_loaded': bool(len(yolo_models) > 0),
            'yolo_classes': classes_out,
            'yolo_weights': yolo_weights_label,
        }
        # provide diagnostics if YOLO not loaded
        if not meta['yolo_loaded']:
            meta['yolo_candidates'] = [
                os.path.relpath(p, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
                if os.path.isabs(p) else p for p in (yolo_candidates_tried or [])
            ]
            if yolo_error:
                meta['yolo_error'] = yolo_error
        return meta

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

    def _cam_to_boxes(cam: np.ndarray, w: int, h: int, thr: float = 0.6, min_area_ratio: float = 0.002):
        cam_resized = cv2.resize(cam, (w, h))
        mask = (cam_resized >= thr).astype(np.uint8) * 255
        # морфология для сглаживания шума
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        except Exception:
            pass
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        min_area = int(min_area_ratio * w * h)
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw * bh < min_area:
                continue
            boxes.append([int(x), int(y), int(x + bw), int(y + bh)])
        return boxes

    def _rust_color_boxes(img_np: np.ndarray, min_area_ratio: float = 0.0012):
        h, w = img_np.shape[:2]
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        # Маска 1: оранжево-коричневые оттенки
        lower1 = np.array([5, 40, 25], dtype=np.uint8)
        upper1 = np.array([30, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        # Маска 2: тёмная ржавчина (низкая яркость/насыщенность)
        lower2 = np.array([0, 20, 10], dtype=np.uint8)
        upper2 = np.array([20, 200, 160], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        # Маска 3: по LAB (красно-жёлтые области)
        a = lab[:, :, 1]
        bch = lab[:, :, 2]
        lab_comb = cv2.merge([a, bch, a])
        mask3 = cv2.inRange(lab_comb, np.array([135, 135, 0], np.uint8), np.array([255, 255, 255], np.uint8))
        mask = cv2.bitwise_or(mask, mask3)
        try:
            k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1, iterations=2)
            k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k2, iterations=1)
        except Exception:
            pass
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        min_area = int(min_area_ratio * w * h)
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw * bh < min_area:
                continue
            boxes.append([int(x), int(y), int(x + bw), int(y + bh)])
        return boxes

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

    @app.post('/api/heatmap_boxes')
    async def heatmap_boxes(head: str = 'dirty', file: UploadFile = File(...), thr: float = 0.6, min_area_ratio: float = 0.002):
        if file.content_type not in {'image/jpeg', 'image/png', 'image/bmp'}:
            raise HTTPException(status_code=400, detail='Неподдерживаемый тип файла (разрешены JPEG/PNG/BMP)')
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert('RGB')
        w, h = img.size
        x = t(img).unsqueeze(0).to(device)
        if head == 'dirty':
            cam = _compute_cam(x, 'clean')
            cam = 1.0 - cam
        else:
            cam = _compute_cam(x, 'damaged')
        boxes = _cam_to_boxes(cam, w, h, float(thr), float(min_area_ratio))
        label = 'грязь' if head == 'dirty' else 'повреждение'
        return JSONResponse({'head': head, 'boxes': boxes, 'label': label})
    @app.post('/api/heatmap_boxes/', include_in_schema=False)
    async def heatmap_boxes_slash(head: str = 'dirty', file: UploadFile = File(...), thr: float = 0.6, min_area_ratio: float = 0.002):
        return await heatmap_boxes(head=head, file=file, thr=thr, min_area_ratio=min_area_ratio)

    @app.post('/api/detect')
    async def detect(file: UploadFile = File(...), conf: float = 0.05, iou: float = 0.45):
        if not yolo_models:
            raise HTTPException(status_code=503, detail='Детектор не загружен. Передайте путь(и) к YOLO весам при запуске.')
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert('RGB')
        img_np = np.array(img)

        # 1) Выбираем модель, у которой есть класс 'car' (авто) — приоритет весам с "car"/"dent" в названии
        car_model = None
        car_model_idx = -1
        for i_m, m in enumerate(yolo_models):
            try:
                names = getattr(m.model, 'names', {}) if hasattr(m.model, 'names') else {}
                has_car = False
                for i_n in range(len(names)):
                    if _friendly_class(names[i_n]) == 'авто':
                        has_car = True
                        break
                if not has_car:
                    continue
                # предпочитаем веса, в названии которых есть 'car' или 'dent'
                is_preferred = False
                try:
                    wbase = os.path.basename(yolo_weights_list[i_m]).lower()
                    if ('car' in wbase) or ('dent' in wbase):
                        is_preferred = True
                except Exception:
                    pass
                if car_model is None or is_preferred:
                    car_model = m
                    car_model_idx = i_m
            except Exception:
                continue

        def _run_yolo(mdl, image, conf_val: float):
            try:
                return mdl.predict(image, conf=conf_val, iou=iou, verbose=False, augment=True)
            except Exception:
                return None

        dets: list[dict] = []

        # 2) Если нашли модель с классом авто — сначала ищем автомобили
        car_boxes: list[tuple[list[float], float]] = []  # ([x1,y1,x2,y2], score)
        if car_model is not None:
            conf_car = max(0.15, float(conf))
            res = _run_yolo(car_model, img, conf_car)
            if res:
                r = res[0]
                boxes = getattr(r, 'boxes', None)
                names = getattr(r, 'names', {}) if hasattr(r, 'names') and isinstance(r.names, dict) else {}
                if boxes is not None:
                    for i_b in range(len(boxes)):
                        b = boxes[i_b]
                        cls_id = int(b.cls.item())
                        raw_name = names.get(cls_id) if isinstance(names, dict) else None
                        if _friendly_class(raw_name) == 'авто':
                            xyxy = b.xyxy[0].tolist()
                            score = float(b.conf.item())
                            car_boxes.append(([float(v) for v in xyxy], score))
            # если машин нет — пробуем пониже порог
            if not car_boxes and conf_car > 0.05:
                res = _run_yolo(car_model, img, 0.05)
                if res:
                    r = res[0]
                    boxes = getattr(r, 'boxes', None)
                    names = getattr(r, 'names', {}) if hasattr(r, 'names') and isinstance(r.names, dict) else {}
                    if boxes is not None:
                        for i_b in range(len(boxes)):
                            b = boxes[i_b]
                            cls_id = int(b.cls.item())
                            raw_name = names.get(cls_id) if isinstance(names, dict) else None
                            if _friendly_class(raw_name) == 'авто':
                                xyxy = b.xyxy[0].tolist()
                                score = float(b.conf.item())
                                car_boxes.append(([float(v) for v in xyxy], score))

        def _clip(v, lo, hi):
            return max(lo, min(hi, v))

        # 3) Если машины найдены — детектим дефекты внутри каждой ROI, иначе — глобальная детекция как раньше
        if car_boxes:
            # добавим боксы машины тоже в выдачу
            for (cb, sc) in car_boxes:
                dets.append({'box': cb, 'class_id': -1, 'class': 'авто', 'score': float(sc)})
            H, W = img_np.shape[:2]
            for (cb, _) in car_boxes:
                x1, y1, x2, y2 = [int(v) for v in cb]
                # паддинг 5% от размера бокса
                pad = int(0.05 * max(x2 - x1, y2 - y1))
                xx1 = _clip(x1 - pad, 0, W - 1)
                yy1 = _clip(y1 - pad, 0, H - 1)
                xx2 = _clip(x2 + pad, 0, W - 1)
                yy2 = _clip(y2 + pad, 0, H - 1)
                if xx2 <= xx1 or yy2 <= yy1:
                    continue
                crop_np = img_np[yy1:yy2, xx1:xx2]
                crop_img = Image.fromarray(crop_np)

                roi_dets: list[dict] = []
                # запускаем все модели и собираем дефекты, исключая класс авто
                for m in yolo_models:
                    res = _run_yolo(m, crop_img, float(conf))
                    if not res:
                        # пробуем понизить конф, если пусто
                        if conf > 0.02:
                            res = _run_yolo(m, crop_img, 0.02)
                    if not res:
                        continue
                    r = res[0]
                    boxes = getattr(r, 'boxes', None)
                    names = getattr(r, 'names', {}) if hasattr(r, 'names') and isinstance(r.names, dict) else {}
                    if boxes is not None:
                        for i_b in range(len(boxes)):
                            b = boxes[i_b]
                            cls_id = int(b.cls.item())
                            raw_name = names.get(cls_id) if isinstance(names, dict) else None
                            friendly = _friendly_class(raw_name)
                            if friendly == 'авто':
                                continue
                            xyxy = b.xyxy[0].tolist()
                            gx1 = float(xyxy[0]) + xx1
                            gy1 = float(xyxy[1]) + yy1
                            gx2 = float(xyxy[2]) + xx1
                            gy2 = float(xyxy[3]) + yy1
                            score = float(b.conf.item())
                            roi_dets.append({'box': [gx1, gy1, gx2, gy2], 'class_id': cls_id, 'class': friendly, 'score': score})

                # Доп. попытка найти ржавчину: низкий conf только для ржавчины
                def _is_rust_label(lbl: Optional[str]) -> bool:
                    return (_friendly_class(lbl) == 'ржавчина') if lbl else False
                if not any((d.get('class') == 'ржавчина') for d in roi_dets):
                    for m in yolo_models:
                        res = _run_yolo(m, crop_img, 0.01)
                        if not res:
                            continue
                        r = res[0]
                        boxes = getattr(r, 'boxes', None)
                        names = getattr(r, 'names', {}) if hasattr(r, 'names') and isinstance(r.names, dict) else {}
                        if boxes is not None:
                            for i_b in range(len(boxes)):
                                b = boxes[i_b]
                                cls_id = int(b.cls.item())
                                raw_name = names.get(cls_id) if isinstance(names, dict) else None
                                if _is_rust_label(raw_name):
                                    xyxy = b.xyxy[0].tolist()
                                    gx1 = float(xyxy[0]) + xx1
                                    gy1 = float(xyxy[1]) + yy1
                                    gx2 = float(xyxy[2]) + xx1
                                    gy2 = float(xyxy[3]) + yy1
                                    score = float(b.conf.item())
                                    roi_dets.append({'box': [gx1, gy1, gx2, gy2], 'class_id': cls_id, 'class': 'ржавчина', 'score': score})
                # Цветовой фолбэк по ржавчине в ROI
                if not any((d.get('class') == 'ржавчина') for d in roi_dets):
                    rust_boxes = _rust_color_boxes(crop_np)
                    for b in rust_boxes:
                        gx1 = float(b[0]) + xx1
                        gy1 = float(b[1]) + yy1
                        gx2 = float(b[2]) + xx1
                        gy2 = float(b[3]) + yy1
                        roi_dets.append({'box': [gx1, gy1, gx2, gy2], 'class_id': -1, 'class': 'ржавчина', 'score': 0.25})

                dets.extend(roi_dets)
        else:
            # fallback: глобальная детекция (как раньше)
            for m in yolo_models:
                dets_m: list[dict] = []
                res = _run_yolo(m, img, float(conf))
                if res:
                    r = res[0]
                    boxes = getattr(r, 'boxes', None)
                    if boxes is not None:
                        for i in range(len(boxes)):
                            b = boxes[i]
                            xyxy = b.xyxy[0].tolist()
                            cls_id = int(b.cls.item())
                            score = float(b.conf.item())
                            raw_name = r.names.get(cls_id) if hasattr(r, 'names') and isinstance(r.names, dict) else None
                            dets_m.append({'box': [float(v) for v in xyxy], 'class_id': cls_id, 'class': _friendly_class(raw_name), 'score': score})
                if not dets_m and (conf is None or conf > 0.02):
                    res = _run_yolo(m, img, 0.02)
                    if res:
                        r = res[0]
                        boxes = getattr(r, 'boxes', None)
                        if boxes is not None:
                            for i in range(len(boxes)):
                                b = boxes[i]
                                xyxy = b.xyxy[0].tolist()
                                cls_id = int(b.cls.item())
                                score = float(b.conf.item())
                                raw_name = r.names.get(cls_id) if hasattr(r, 'names') and isinstance(r.names, dict) else None
                                dets_m.append({'box': [float(v) for v in xyxy], 'class_id': cls_id, 'class': _friendly_class(raw_name), 'score': score})
                dets.extend(dets_m)

            # добивка ржавчины глобально
            def _is_rust_label(lbl: Optional[str]) -> bool:
                return (_friendly_class(lbl) == 'ржавчина') if lbl else False
            if not any((d.get('class') == 'ржавчина') for d in dets):
                for m in yolo_models:
                    res = _run_yolo(m, img, 0.01)
                    if res:
                        r = res[0]
                        boxes = getattr(r, 'boxes', None)
                        names = getattr(r, 'names', {}) if hasattr(r, 'names') and isinstance(r.names, dict) else {}
                        if boxes is not None:
                            for i in range(len(boxes)):
                                b = boxes[i]
                                cls_id = int(b.cls.item())
                                raw_name = names.get(cls_id) if isinstance(names, dict) else None
                                if _is_rust_label(raw_name):
                                    xyxy = b.xyxy[0].tolist()
                                    score = float(b.conf.item())
                                    dets.append({'box': [float(v) for v in xyxy], 'class_id': cls_id, 'class': 'ржавчина', 'score': score})
            if not any((d.get('class') == 'ржавчина') for d in dets):
                rust_boxes = _rust_color_boxes(img_np)
                for b in rust_boxes:
                    dets.append({'box': [float(v) for v in b], 'class_id': -1, 'class': 'ржавчина', 'score': 0.25})

        # NMS по классам
        def iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = area_a + area_b - inter + 1e-6
            return inter / union
        nms_out: list[dict] = []
        by_cls: dict[str, list[dict]] = {}
        for d in dets:
            c = (d.get('class') or '').lower()
            by_cls.setdefault(c, []).append(d)
        for c, arr in by_cls.items():
            arr.sort(key=lambda x: x.get('score', 0.0), reverse=True)
            kept: list[dict] = []
            for d in arr:
                db = d['box']
                if all((iou(db, k['box']) < 0.5) for k in kept):
                    kept.append(d)
            nms_out.extend(kept)

        # Собираем список классов
        classes_out = None
        try:
            s = set()
            for m in yolo_models:
                names = getattr(m.model, 'names', {})
                for i in range(len(names)):
                    fn = _friendly_class(names[i])
                    if fn:
                        s.add(fn)
            classes_out = sorted(s)
        except Exception:
            classes_out = None
        return JSONResponse({'detections': nms_out, 'classes': classes_out})
    @app.post('/api/detect/', include_in_schema=False)
    async def detect_slash(file: UploadFile = File(...), conf: float = 0.05, iou: float = 0.45):
        return await detect(file=file, conf=conf, iou=iou)

    @app.post('/api/detect_image')
    async def detect_image(file: UploadFile = File(...), conf: float = 0.05, iou: float = 0.45):
        if not yolo_models:
            raise HTTPException(status_code=503, detail='Детектор не загружен. Передайте путь(и) к YOLO весам при запуске.')
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert('RGB')
        # For visualization use the first model
        m0 = yolo_models[0]
        results = m0.predict(img, conf=conf, iou=iou, verbose=False, augment=True)
        if not results:
            raise HTTPException(status_code=500, detail='YOLO inference failed')
        r = results[0]
        # if no boxes, retry with lower conf
        try:
            if r.boxes is not None and len(r.boxes) == 0 and (conf is None or conf > 0.035):
                results = m0.predict(img, conf=0.03, iou=iou, verbose=False, augment=True)
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
    async def detect_image_slash(file: UploadFile = File(...), conf: float = 0.05, iou: float = 0.45):
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
