import io
import os
from dataclasses import dataclass

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms
import tyro

from ..model import MultiHeadResNet


def build_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def make_app(ckpt: str, image_size: int = 224, device: str = 'cpu'):
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model = MultiHeadResNet(pretrained=False)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state['model'])
    model.to(device).eval()

    t = build_transform(image_size)

    app = FastAPI(title='Car Condition Web')

    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    app.mount('/static', StaticFiles(directory=static_dir), name='static')

    @app.get('/', response_class=HTMLResponse)
    def index():
        with open(os.path.join(static_dir, 'index.html'), 'r', encoding='utf-8') as f:
            return f.read()

    @app.get('/health')
    def health():
        return {'status': 'ok'}

    @app.post('/api/predict')
    async def predict(file: UploadFile = File(...)):
        if file.content_type not in {'image/jpeg', 'image/png', 'image/bmp'}:
            raise HTTPException(status_code=400, detail='Unsupported file type')
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert('RGB')
        x = t(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pc, pd = model.predict_proba(x)
            pc = float(pc.item())
            pd = float(pd.item())
        return JSONResponse({
            'clean_prob': pc,
            'dirty_prob': 1 - pc,
            'damaged_prob': pd,
            'intact_prob': 1 - pd,
            'pred_clean': 'clean' if pc >= 0.5 else 'dirty',
            'pred_damage': 'damaged' if pd >= 0.5 else 'intact',
        })

    return app


@dataclass
class WebConfig:
    ckpt: str = 'checkpoints/resnet18_multitask.pt'
    host: str = '127.0.0.1'
    port: int = 8000
    image_size: int = 224
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(cfg: WebConfig):
    import uvicorn
    app = make_app(cfg.ckpt, cfg.image_size, cfg.device)
    uvicorn.run(app, host=cfg.host, port=cfg.port)


if __name__ == '__main__':
    cfg = tyro.cli(WebConfig)
    main(cfg)
