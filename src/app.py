import gradio as gr
import torch
from dataclasses import dataclass
from PIL import Image
from torchvision import transforms
import tyro

from .model import MultiHeadResNet


def build_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def make_interface(ckpt: str, image_size: int = 224, device: str = 'cpu'):
    model = MultiHeadResNet(pretrained=False)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state['model'])
    model.to(device).eval()
    t = build_transform(image_size)

    def infer(img: Image.Image):
        x = t(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pc, pd = model.predict_proba(x)
            pc = pc.item()
            pd = pd.item()
        clean_label = { 'clean': pc, 'dirty': 1 - pc }
        damage_label = { 'damaged': pd, 'intact': 1 - pd }
        return clean_label, damage_label

    demo = gr.Interface(
        fn=infer,
        inputs=gr.Image(type='pil'),
        outputs=[
            gr.Label(num_top_classes=2, label='Clean vs Dirty'),
            gr.Label(num_top_classes=2, label='Damaged vs Intact')
        ],
        examples=None,
        title='Car Condition Classifier',
        description='Загрузите фото авто, модель оценит чистоту и повреждения.'
    )
    return demo


@dataclass
class AppConfig:
    ckpt: str
    image_size: int = 224
    server_port: int = 7860
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(cfg: AppConfig):
    demo = make_interface(cfg.ckpt, cfg.image_size, cfg.device)
    demo.launch(server_port=cfg.server_port)


if __name__ == '__main__':
    cfg = tyro.cli(AppConfig)
    main(cfg)
