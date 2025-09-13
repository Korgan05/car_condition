import os
import csv
import glob
import argparse
import torch
from PIL import Image
from torchvision import transforms

from src.model import MultiHeadResNet


def build_transform(size=224):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict_path(model, device, t, path):
    img = Image.open(path).convert('RGB')
    x = t(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pc, pd = model.predict_proba(x)
        pc = float(pc.item()); pd = float(pd.item())
    return pc, pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--folder', required=True)
    ap.add_argument('--out', default='predictions.csv')
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    model = MultiHeadResNet(pretrained=False)
    state = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(state['model'])
    model.to(args.device).eval()

    t = build_transform(args.image_size)
    patterns = ['*.jpg','*.jpeg','*.png','*.bmp']
    paths = []
    for p in patterns:
        paths.extend(glob.glob(os.path.join(args.folder, p)))

    rows = [("path","clean_prob","dirty_prob","damaged_prob","intact_prob","pred_clean","pred_damage")]
    for p in paths:
        pc, pd = predict_path(model, args.device, t, p)
        rows.append((p, pc, 1-pc, pd, 1-pd, 'clean' if pc>=0.5 else 'dirty', 'damaged' if pd>=0.5 else 'intact'))

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerows(rows)
    print('Saved predictions to', args.out)

if __name__ == '__main__':
    main()
