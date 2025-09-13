import os
import csv
import argparse
from glob import glob

"""
Конвертация выгрузки изображений в labels.csv.
Простой подход: классы определяются из имени файла/подпапки.
Ожидается структура:
  root/
    clean/*.jpg|png
    dirty/*.jpg|png
    damaged/*.jpg|png
    intact/*.jpg|png
Можно использовать любые подмножества; метки аггрегируются так:
  clean = 1, если файл в clean; 0 если в dirty; иначе неизвестно -> пропускаем
  damaged = 1, если файл в damaged; 0 если в intact; иначе неизвестно -> пропускаем
В файл попадут только изображения, у которых определены обе метки.
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Корневая папка с изображениями по подпапкам')
    ap.add_argument('--out', default='data/labels.csv')
    ap.add_argument('--exts', nargs='+', default=['.jpg','.jpeg','.png','.bmp'])
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    cats = {
        'clean': set(),
        'dirty': set(),
        'damaged': set(),
        'intact': set(),
    }

    def collect(cat):
        for ext in args.exts:
            for p in glob(os.path.join(root, cat, f'*{ext}')):
                cats[cat].add(os.path.abspath(p))

    for c in cats.keys():
        collect(c)

    all_paths = set.union(*cats.values())
    rows = [("filepath","clean","damaged")]

    for p in sorted(all_paths):
        c = 1 if p in cats['clean'] else 0 if p in cats['dirty'] else None
        d = 1 if p in cats['damaged'] else 0 if p in cats['intact'] else None
        if c is None or d is None:
            continue
        # путь относительный к корню репо
        rel = os.path.relpath(p, os.path.dirname(os.path.dirname(__file__)))
        rows.append((rel, c, d))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerows(rows)
    print(f'Saved {len(rows)-1} rows to {args.out}')

if __name__ == '__main__':
    main()
