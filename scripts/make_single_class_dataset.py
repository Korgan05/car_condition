import argparse
import shutil
from pathlib import Path
import yaml


def load_yaml(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def filter_label_file(src_txt: Path, dst_txt: Path, keep_idx: int) -> int:
    kept = 0
    if not src_txt.exists():
        # No labels -> background image
        return kept
    lines_out = []
    with src_txt.open('r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cls = int(float(parts[0]))
            except ValueError:
                continue
            if cls == keep_idx:
                parts[0] = '0'  # remap to 0 for single-class
                lines_out.append(' '.join(parts))
                kept += 1
    if lines_out:
        dst_txt.parent.mkdir(parents=True, exist_ok=True)
        with dst_txt.open('w', encoding='utf-8') as f:
            f.write('\n'.join(lines_out) + '\n')
    else:
        # If nothing kept, write empty file or skip; YOLO handles missing labels too.
        # We'll skip creating a file to reduce clutter.
        pass
    return kept


def copy_image(src_img: Path, dst_img: Path):
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, dst_img)


def build_single_class_dataset(src_dir: Path, dst_dir: Path, keep_class_name: str):
    src_yaml = src_dir / 'data.yaml'
    assert src_yaml.exists(), f"data.yaml not found at {src_yaml}"
    data = load_yaml(src_yaml)

    names = data.get('names') or []
    if isinstance(names, dict):
        # sometimes names are indexed dict
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    try:
        keep_idx = names.index(keep_class_name)
    except ValueError:
        raise SystemExit(f"Class '{keep_class_name}' not found in names: {names}")

    # Resolve split image directories (handle Roboflow-style '../train/images')
    def resolve_split(split_key: str):
        if split_key == 'train':
            p = data.get('train')
        elif split_key in ('val', 'valid'):
            p = data.get('val') or data.get('valid')
        else:
            p = data.get('test')
        if not p:
            return None
        p_str = str(p)
        # If path starts with '../', interpret relative to src_dir (dataset root)
        if p_str.startswith('../'):
            p_rel = p_str[3:]
            return (src_dir / p_rel).resolve()
        # Else, join relative to data.yaml location
        return (src_yaml.parent / p_str).resolve()

    splits = {
        'train': resolve_split('train'),
        'valid': resolve_split('val'),
        'test': resolve_split('test'),
    }

    # Create destination structure
    for split, img_dir in splits.items():
        if not img_dir or not img_dir.exists():
            continue
        dst_img_dir = dst_dir / split / 'images'
        dst_lbl_dir = dst_dir / split / 'labels'
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img in img_dir.glob('*.*'):
            if img.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                continue
            copy_image(img, dst_img_dir / img.name)
            # label path: replace images -> labels and extension .txt
            rel = img.relative_to(img_dir)
            lbl_src = (img_dir.parent / 'labels' / rel).with_suffix('.txt')
            lbl_dst = (dst_lbl_dir / rel).with_suffix('.txt')
            filter_label_file(lbl_src, lbl_dst, keep_idx)

    # Write new data.yaml
    new_yaml = {
        'train': '../train/images',
        'val': '../valid/images',
        'test': '../test/images',
        'nc': 1,
        'names': [keep_class_name],
        'source': str(src_dir.resolve()),
        'note': f"Generated single-class dataset for '{keep_class_name}'",
    }
    save_yaml(new_yaml, dst_dir / 'data.yaml')


def main():
    ap = argparse.ArgumentParser(description='Make single-class YOLO dataset from multi-class dataset')
    ap.add_argument('--src', required=True, help='Path to source dataset folder containing data.yaml')
    ap.add_argument('--dst', required=True, help='Path to output dataset folder')
    ap.add_argument('--class', dest='klass', required=True, help='Class name to keep, e.g., dent')
    args = ap.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    build_single_class_dataset(src_dir, dst_dir, args.klass)
    print(f"Single-class dataset created at: {dst_dir}")


if __name__ == '__main__':
    main()
