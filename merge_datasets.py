"""
merge_datasets.py

Combines multiple Roboflow YOLO-format exports (each with its own class
ordering/names) into a single dataset with one canonical class list, so you
can train one YOLOv8 model on all of them together.

Usage:
    python merge_datasets.py

Expects:
    data/raw/<dataset_1>/{train,valid,test}/{images,labels}/  + data.yaml
    data/raw/<dataset_2>/{train,valid,test}/{images,labels}/  + data.yaml
    ...

Produces:
    data/merged/{train,valid,test}/{images,labels}/
    data/merged/data.yaml
"""

import os
import re
import shutil
import yaml

RAW_DIR = "data/raw"
OUT_DIR = "data/merged"

# The canonical class list your final model will use.
# Edit this if you want to include coins, or drop rare denominations.
CANONICAL_CLASSES = [
    "10", "20", "50", "100", "200", "500", "2000",
]

# Maps messy source label variants -> canonical class name.
# Add entries here as you discover new naming conventions in downloaded sets.
ALIAS_MAP = {
    "10": "10", "rs10": "10", "10 rupee note": "10", "ten rupees note": "10",
    "20": "20", "rs20": "20", "20 rupee note": "20", "twenty rupees note": "20",
    "50": "50", "rs50": "50", "50 rupee note": "50", "fifty rupees note": "50",
    "100": "100", "rs100": "100", "100 rupee note": "100", "hundred rupees note": "100",
    "200": "200", "rs200": "200", "200 rupee note": "200", "two hundred rupees note": "200",
    "500": "500", "rs500": "500", "500 rupee note": "500", "five hundred rupees note": "500",
    "2000": "2000", "rs2000": "2000", "2000 rupee note": "2000", "two thousand rupees note": "2000",
}

# Per-dataset overrides for datasets whose data.yaml class "names" are
# meaningless (e.g. just '0','1','2'...) instead of readable denomination
# labels. Match is done by substring against the dataset's folder name
# (case-insensitive). Value is a dict mapping the ORIGINAL class index
# (as it appears in that dataset's label files, i.e. position in its
# data.yaml names list) -> canonical denomination string.
#
# Fill this in once you've identified the mapping with crop_by_class.py, e.g.:
#   "INDIAN CURRENCY DETECTION.v4i.yolov8": {
#       0: "100", 1: "10", 2: "200", 3: "20", 4: "2000", 5: "50", 6: "500",
#   },
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
        # try to pull a bare number out (handles things like "note-500", "500rs")
        digits = re.findall(r"\d+", key)
        if digits and digits[0] in CANONICAL_CLASSES:
            mapped = digits[0]
    if mapped is None or mapped not in CANONICAL_CLASSES:
        return None
    return CANONICAL_CLASSES.index(mapped)


def ensure_out_dirs():
    for split in ["train", "valid", "test"]:
        os.makedirs(f"{OUT_DIR}/{split}/images", exist_ok=True)
        os.makedirs(f"{OUT_DIR}/{split}/labels", exist_ok=True)


def remap_labels(label_path, src_classes, dst_path, index_override=None):
    """Rewrite a YOLO label file, remapping class indices to canonical set.
    Drops any box whose class doesn't map to a canonical class (e.g. 'None').

    If index_override is given (dict of original_index -> canonical denom
    string), it's used directly instead of trying to match src_classes by
    name -- needed for datasets whose class names are just '0','1','2'..."""
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


def process_dataset(dataset_dir):
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"  ! skipping {dataset_dir}, no data.yaml found")
        return 0, 0

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    src_classes = cfg.get("names", [])
    if isinstance(src_classes, dict):  # some exports use {0: 'name', ...}
        src_classes = [src_classes[i] for i in sorted(src_classes.keys())]

    dataset_name = os.path.basename(dataset_dir.rstrip("/").rstrip("\\"))
    index_override = find_override(dataset_name)
    if index_override:
        print(f"  using class-index override for '{dataset_name}': {index_override}")
    kept, skipped = 0, 0

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
            dst_lbl = f"{OUT_DIR}/{split}/labels/{new_stem}.txt"

            has_boxes = remap_labels(lbl_path, src_classes, dst_lbl, index_override=index_override)
            if has_boxes:
                shutil.copy2(
                    os.path.join(img_dir, img_name),
                    f"{OUT_DIR}/{split}/images/{new_stem}{ext}",
                )
                kept += 1
            else:
                skipped += 1

    return kept, skipped


def write_merged_yaml():
    cfg = {
        "path": os.path.abspath(OUT_DIR),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
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

    ensure_out_dirs()
    total_kept, total_skipped = 0, 0

    for entry in sorted(os.listdir(RAW_DIR)):
        dataset_dir = os.path.join(RAW_DIR, entry)
        if not os.path.isdir(dataset_dir):
            continue
        print(f"Processing {entry} ...")
        kept, skipped = process_dataset(dataset_dir)
        print(f"  kept {kept} images, skipped {skipped} (no canonical labels)")
        total_kept += kept
        total_skipped += skipped

    write_merged_yaml()
    print(f"\nDone. {total_kept} images merged into {OUT_DIR}/ "
          f"({total_skipped} skipped). Wrote {OUT_DIR}/data.yaml")


if __name__ == "__main__":
    main()
