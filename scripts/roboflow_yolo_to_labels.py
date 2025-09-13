import os
import csv
import argparse
from glob import glob
from typing import Dict, List, Set

"""
Конвертация Roboflow YOLOv8 датасета (детекция/сегментация) в labels.csv для нашего классификатора.

Логика:
  damaged = 1, если у изображения есть хотя бы одна аннотация из заданных классов повреждений
            (по умолчанию: rust/scratch/dent + частые опечатки)
  damaged = 0, если аннотаций этих классов нет (класс 'car' игнорируется)
  clean = 1 - damaged (временный прокси для второй головы)

Ожидаемая структура YOLOv8 экспорта Roboflow:
  root/
    data.yaml (или dataset.yaml) с полем 'names'
    train/images/*.jpg|png, train/labels/*.txt
    valid/images/*,       valid/labels/*   (или 'val')
    test/images/*,        test/labels/*

Запуск:
  python -m scripts.roboflow_yolo_to_labels --root path/to/yolov8_export --out data/labels.csv
"""


def find_yaml(root: str) -> str:
    for name in ["data.yaml", "dataset.yaml", "data.yml", "dataset.yml"]:
        p = os.path.join(root, name)
        if os.path.exists(p):
            return p
    # как fallback — возьмём первый .y*ml
    ys = glob(os.path.join(root, "*.y*ml"))
    return ys[0] if ys else ""


def load_names(yaml_path: str) -> Dict[int, str]:
    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    names = y.get("names")
    if isinstance(names, dict):
        # уже id->name
        return {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    else:
        raise ValueError("Не удалось прочитать список классов из YAML: 'names'")


def list_images(split_dir: str, exts=(".jpg", ".jpeg", ".png", ".bmp")) -> List[str]:
    img_dir = os.path.join(split_dir, "images")
    if not os.path.isdir(img_dir):
        return []
    files: List[str] = []
    for e in exts:
        files.extend(glob(os.path.join(img_dir, f"*{e}")))
    return files


def label_path_for_image(img_path: str) -> str:
    # .../split/images/xxx.jpg -> .../split/labels/xxx.txt
    d, fn = os.path.split(img_path)
    split_dir = os.path.dirname(d)
    lbl_dir = os.path.join(split_dir, "labels")
    base, _ = os.path.splitext(fn)
    return os.path.join(lbl_dir, base + ".txt")


def has_damage(lbl_path: str, id2name: Dict[int, str], damage_names: Set[str]) -> bool:
    if not os.path.exists(lbl_path):
        return False
    try:
        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    except Exception:
        return False
    for ln in lines:
        parts = ln.split()
        if not parts:
            continue
        try:
            cid = int(float(parts[0]))  # YOLO: class_id cx cy w h [...]
        except Exception:
            continue
        name = id2name.get(cid, "").lower()
        if name in damage_names:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Путь к распакованному YOLOv8 экспорту Roboflow")
    ap.add_argument("--out", default="data/labels.csv")
    ap.add_argument("--damage-names", nargs="+",
                    default=[
                        "rust", "scratch", "dent", "damage", "damaged",
                        # частые опечатки в названиях классов
                        "dunt", "scrach", "scracth",
                    ])
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    yaml_path = find_yaml(root)
    if not yaml_path:
        raise FileNotFoundError("Не найден data.yaml/dataset.yaml в корне экспорта")
    id2name = load_names(yaml_path)

    # множества имён классов (в нижнем регистре) для детекта повреждений
    dmg_names = set(n.lower() for n in args.damage_names)
    ignore_names = {"car"}

    rows = [("filepath", "clean", "damaged")]
    repo_root = os.path.dirname(os.path.dirname(__file__))

    # поддержка валид/валид/тест (некоторые экспорты используют 'val')
    split_names = ["train", "valid", "val", "test"]
    total_imgs = 0
    total_dmg = 0
    for sp in split_names:
        sp_dir = os.path.join(root, sp)
        if not os.path.isdir(sp_dir):
            continue
        imgs = list_images(sp_dir)
        for ip in imgs:
            total_imgs += 1
            lp = label_path_for_image(ip)
            is_dmg = has_damage(lp, id2name, dmg_names)
            total_dmg += int(is_dmg)
            clean = 0 if is_dmg else 1
            rel = os.path.relpath(ip, repo_root)
            rows.append((rel.replace("\\", "/"), clean, int(is_dmg)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerows(rows)
    print(f"Saved {len(rows)-1} rows to {args.out} | images={total_imgs}, damaged={total_dmg}")


if __name__ == "__main__":
    main()
