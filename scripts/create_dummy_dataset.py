import os
import csv
import random
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, 'data', 'raw')
LABELS = os.path.join(ROOT, 'data', 'labels.csv')

os.makedirs(DATA_DIR, exist_ok=True)

random.seed(0)

rows = [("filepath","clean","damaged")]

for i in range(60):
    w, h = 256, 256
    # чистые — светлые; грязные — более тёмные/зашумлённые
    clean = random.randint(0,1)
    damaged = random.randint(0,1)

    bg = 220 if clean==1 else 100
    img = Image.new('RGB', (w,h), (bg,bg,bg))
    dr = ImageDraw.Draw(img)

    # повреждения — рисуем красные/чёрные линии
    if damaged:
        for _ in range(random.randint(3,8)):
            x1, y1 = random.randint(0,w-1), random.randint(0,h-1)
            x2, y2 = random.randint(0,w-1), random.randint(0,h-1)
            dr.line((x1,y1,x2,y2), fill=(200,0,0), width=random.randint(1,4))

    fname = f'dummy_{i:03d}.png'
    path = os.path.join(DATA_DIR, fname)
    img.save(path)

    # путь относительно корня проекта
    rel = os.path.relpath(path, ROOT)
    rows.append((rel, clean, damaged))

with open(LABELS, 'w', newline='', encoding='utf-8') as f:
    wr = csv.writer(f)
    wr.writerows(rows)

print('Dummy dataset created at', DATA_DIR)
print('Labels:', LABELS)
