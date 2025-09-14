import argparse
import json
import os
import shutil
from typing import Dict, List, Tuple


DEFAULT_CLASSES = [
    # Keep order consistent with our YOLO weights metadata
    "car",
    "dunt",      # dent/вмятина (в проекте встречается опечатка 'dunt')
    "rust",      # ржавчина
    "scracth",   # царапина (в проекте встречается опечатка 'scracth')
]


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _load_boxes_from_json(j: Dict) -> Tuple[str, int, int, List[Dict]]:
    """
    Supports two shapes the user sent:
    1) Top-level Roboflow record with annotations -> annotations[<key>]['converted'] (string JSON)
    2) Direct object with {key,width,height,boxes:[{label,x,y,width,height}]}
    Returns: (image_filename, W, H, boxes)
    """
    if "boxes" in j and "key" in j and "width" in j and "height" in j:
        return j["key"], int(float(j["width"])), int(float(j["height"])), j["boxes"]

    # Try Roboflow record shape
    ann = j.get("annotations") or {}
    if ann:
        # Take first annotation record
        _, rec = next(iter(ann.items()))
        conv = rec.get("converted")
        if conv:
            inner = json.loads(conv)
            return inner["key"], int(inner["width"]), int(inner["height"]), inner["boxes"]

    raise ValueError("Unsupported JSON shape. Expect either direct boxes object or Roboflow record with 'converted'.")


def _to_yolo_xywhn(box: Dict, W: int, H: int) -> Tuple[float, float, float, float]:
    # Box is {label, x, y, width, height} with absolute values in pixels, x,y are top-left
    x = float(box["x"]) if isinstance(box["x"], str) else float(box["x"])
    y = float(box["y"]) if isinstance(box["y"], str) else float(box["y"])
    w = float(box["width"]) if isinstance(box["width"], str) else float(box["width"])
    h = float(box["height"]) if isinstance(box["height"], str) else float(box["height"])
    # Convert to center-x, center-y
    cx = x + w / 2.0
    cy = y + h / 2.0
    # Normalize
    return cx / W, cy / H, w / W, h / H


def main():
    ap = argparse.ArgumentParser(description="Convert Roboflow record JSON to YOLOv8 labels (.txt)")
    ap.add_argument("json_path", help="Path to Roboflow JSON file")
    ap.add_argument("out_dir", help="Output directory (labels will be written here)")
    ap.add_argument("--images-root", dest="images_root", help="Folder with source images to optionally copy", default=None)
    ap.add_argument("--copy-images", dest="copy_images", action="store_true", help="Copy image into out_dir/images")
    ap.add_argument("--classes", nargs="*", help="Override classes order. Default: %s" % DEFAULT_CLASSES)
    args = ap.parse_args()

    classes = args.classes or DEFAULT_CLASSES
    labels_dir = os.path.join(args.out_dir, "labels")
    images_out = os.path.join(args.out_dir, "images")
    _ensure_dir(labels_dir)
    if args.copy_images:
        _ensure_dir(images_out)

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    key, W, H, boxes = _load_boxes_from_json(data)
    base = os.path.splitext(os.path.basename(key))[0]
    txt_path = os.path.join(labels_dir, base + ".txt")

    lines = []
    for b in boxes:
        label = str(b.get("label") or b.get("class") or "").strip()
        if label not in classes:
            # skip unknown labels, or add new class index at the end
            classes.append(label)
        cls_id = classes.index(label)
        x, y, w, h = _to_yolo_xywhn(b, W, H)
        lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    # Save classes.txt for reference
    with open(os.path.join(args.out_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(classes) + "\n")

    # Optionally copy image into out_dir/images
    if args.copy_images:
        if not args.images_root:
            print("--copy-images specified but --images-root not provided; skipping image copy")
        else:
            src_img = os.path.join(args.images_root, key)
            if os.path.exists(src_img):
                _ensure_dir(os.path.dirname(os.path.join(images_out, key)))
                shutil.copy2(src_img, os.path.join(images_out, key))
            else:
                print(f"Image not found to copy: {src_img}")

    print(f"Written: {txt_path}")


if __name__ == "__main__":
    main()
