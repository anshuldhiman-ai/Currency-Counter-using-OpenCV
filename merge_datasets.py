"""
merge_datasets.py

Combines multiple Roboflow YOLO-format exports (each with its own class
ordering/names) into a single dataset with one canonical class list, so you
   can train one YOLOv8 model on all of them together.

It pools all images and labels across all splits (train/valid/test) and performs
a balanced 85% train / 15% validation split to ensure all classes (including 200)
have proper representation in the training set.

Usage:
    python merge_datasets.py
"""

import os
import re
import shutil
import random
import yaml

RAW_DIR = "data/raw"
OUT_DIR = "data/merged"

# The canonical class list your final model will use.
# Added "5" and removed "2000" as requested by the user.
CANONICAL_CLASSES = [
    "5", "10", "20", "50", "100", "200", "500",
]

# Maps messy source label variants -> canonical class name.
ALIAS_MAP = {
    "5": "5", "rs5": "5", "5 rupee note": "5", "five rupees note": "5",
    "10": "10", "rs10": "10", "10 rupee note": "10", "ten rupees note": "10",
    "20": "20", "rs20": "20", "20 rupee note": "20", "twenty rupees note": "20",
    "50": "50", "rs50": "50", "50 rupee note": "50", "fifty rupees note": "50",
    "100": "100", "rs100": "100", "100 rupee note": "100", "hundred rupees note": "100",
    "200": "200", "rs200": "200", "200 rupee note": "200", "two hundred rupees note": "200",
    "500": "500", "rs500": "500", "500 rupee note": "500", "five hundred rupees note": "500",
}

# Per-dataset overrides for datasets whose class indices are meaningless.
DATASET_CLASS_INDEX_OVERRIDES = {
    "INDIAN CURRENCY DETECTION.v4i.yolov8": {
        0: "100",
        1: "10",
        2: "200",
        3: "20",
        4: "500",
        5: "50",
        6: "5",
    },
}


def find_override(dataset_name):
    for key, mapping in DATASET_CLASS_INDEX_OVERRIDES.items():
        if key.lower() in dataset_name.lower():
            return mapping
    return None


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", name.strip().lower())


def canonical_index(name: str):
    key = normalize(name)
    mapped = ALIAS_MAP.get(key)
    if mapped is None:
        digits = re.findall(r"\d+", key)
        if digits and digits[0] in CANONICAL_CLASSES:
            mapped = digits[0]
    if mapped is None or mapped not in CANONICAL_CLASSES:
        return None
    return CANONICAL_CLASSES.index(mapped)


def ensure_out_dirs():
    for split in ["train", "valid"]:
        os.makedirs(f"{OUT_DIR}/{split}/images", exist_ok=True)
        os.makedirs(f"{OUT_DIR}/{split}/labels", exist_ok=True)


def remap_labels(label_path, src_classes, dst_path, index_override=None):
    """Rewrite a YOLO label file, remapping class indices to canonical set."""
    kept_lines = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_idx = int(parts[0])

            if index_override is not None:
                denom = index_override.get(cls_idx)
                new_idx = CANONICAL_CLASSES.index(denom) if denom in CANONICAL_CLASSES else None
            else:
                if cls_idx >= len(src_classes):
                    continue
                src_name = src_classes[cls_idx]
                new_idx = canonical_index(src_name)

            if new_idx is None:
                continue
            kept_lines.append(" ".join([str(new_idx)] + parts[1:]))
            
    if kept_lines:
        with open(dst_path, "w") as f:
            f.write("\n".join(kept_lines) + "\n")
        return True
    return False


def write_merged_yaml():
    cfg = {
        "path": os.path.abspath(OUT_DIR),
        "train": "train/images",
        "val": "valid/images",
        "nc": len(CANONICAL_CLASSES),
        "names": CANONICAL_CLASSES,
    }
    with open(f"{OUT_DIR}/data.yaml", "w") as f:
        yaml.dump(cfg, f, sort_keys=False)


def main():
    if not os.path.isdir(RAW_DIR) or not os.listdir(RAW_DIR):
        print(f"No datasets found in {RAW_DIR}/. Download Roboflow exports "
              f"(YOLOv8 format) and unzip them into subfolders there first.")
        return

    # Reset output directory
    if os.path.exists(OUT_DIR):
        print(f"[*] Removing old merged dataset directory '{OUT_DIR}'...")
        shutil.rmtree(OUT_DIR)
        
    pooled_pairs = []

    # Pool all raw files across all splits
    for entry in sorted(os.listdir(RAW_DIR)):
        dataset_dir = os.path.join(RAW_DIR, entry)
        if not os.path.isdir(dataset_dir):
            continue
            
        yaml_path = os.path.join(dataset_dir, "data.yaml")
        if not os.path.exists(yaml_path):
            print(f"  ! skipping {entry}, no data.yaml found")
            continue

        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
        src_classes = cfg.get("names", [])
        if isinstance(src_classes, dict):
            src_classes = [src_classes[i] for i in sorted(src_classes.keys())]

        dataset_name = os.path.basename(dataset_dir.rstrip("/\\"))
        index_override = find_override(dataset_name)
        
        print(f"[*] Pooling files from {entry}...")
        for split in ["train", "valid", "test"]:
            img_dir = os.path.join(dataset_dir, split, "images")
            lbl_dir = os.path.join(dataset_dir, split, "labels")
            if not os.path.isdir(img_dir):
                continue

            for img_name in os.listdir(img_dir):
                stem, ext = os.path.splitext(img_name)
                lbl_path = os.path.join(lbl_dir, stem + ".txt")
                if not os.path.exists(lbl_path):
                    continue

                new_stem = f"{dataset_name}_{stem}"
                pooled_pairs.append({
                    'src_img': os.path.join(img_dir, img_name),
                    'src_lbl': lbl_path,
                    'new_stem': new_stem,
                    'ext': ext,
                    'index_override': index_override,
                    'src_classes': src_classes
                })

    # Shuffle the pooled dataset to distribute classes evenly
    print(f"[*] Shuffling and splitting pooled dataset (total {len(pooled_pairs)} images)...")
    random.seed(42)  # For reproducible splits
    random.shuffle(pooled_pairs)

    # Split into Train (85%) and Val (15%)
    total = len(pooled_pairs)
    train_size = int(total * 0.85)
    
    train_pairs = pooled_pairs[:train_size]
    val_pairs = pooled_pairs[train_size:]

    ensure_out_dirs()

    # Process and copy files
    def write_split(pairs, split_name):
        kept = 0
        for p in pairs:
            dst_lbl = f"{OUT_DIR}/{split_name}/labels/{p['new_stem']}.txt"
            has_boxes = remap_labels(
                p['src_lbl'], 
                p['src_classes'], 
                dst_lbl, 
                index_override=p['index_override']
            )
            if has_boxes:
                shutil.copy2(p['src_img'], f"{OUT_DIR}/{split_name}/images/{p['new_stem']}{p['ext']}")
                kept += 1
        return kept

    print("[*] Writing train split...")
    train_kept = write_split(train_pairs, "train")
    print(f"[+] Wrote {train_kept} train images.")

    print("[*] Writing validation split...")
    val_kept = write_split(val_pairs, "valid")
    print(f"[+] Wrote {val_kept} validation images.")

    write_merged_yaml()
    print(f"\n[+] Done. Merged dataset written to {OUT_DIR}/.")
    print(f"    Wrote {OUT_DIR}/data.yaml. Total images: {train_kept + val_kept}.")


if __name__ == "__main__":
    main()
