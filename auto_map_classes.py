"""
auto_map_classes.py

Some Roboflow datasets keep the original filename (which encodes the
denomination, e.g. "10IndianRupees_10_jpeg.rf.<hash>.jpg") even though the
class "names" in data.yaml were reduced to meaningless numbers ('0','1'...).

This script:
  1. Scans crops_by_class/class_N/*.jpg (produced by crop_by_class.py)
  2. Extracts the denomination from each filename via regex
  3. Takes a majority vote per class to build class_index -> denomination
  4. Prints the mapping and sanity-checks it (all 7 denominations covered,
     no duplicates)
  5. Automatically patches merge_datasets.py's DATASET_CLASS_INDEX_OVERRIDES
     dict with the result (keeps a .bak backup first)

Usage:
    python auto_map_classes.py "data/raw/INDIAN CURRENCY DETECTION.v4i.yolov8"
"""

import argparse
import os
import re
import shutil
from collections import Counter

CANONICAL = ["10", "20", "50", "100", "200", "500", "2000"]

# Matches "10IndianRupees", "500 Indian Rupees", "2000_Indian_Rupees" etc.
PATTERN = re.compile(r"(\d+)\s*indian\s*rupees", re.IGNORECASE)


def extract_denomination(filename):
    cleaned = filename.replace("_", " ")
    m = PATTERN.search(cleaned)
    if m:
        val = m.group(1)
        return val if val in CANONICAL else None
    return None


def build_mapping(crops_dir):
    mapping = {}
    ambiguous = {}

    class_dirs = sorted(
        [d for d in os.listdir(crops_dir) if d.startswith("class_")],
        key=lambda d: int(d.split("_")[1]),
    )

    for class_dir in class_dirs:
        idx = int(class_dir.split("_")[1])
        folder = os.path.join(crops_dir, class_dir)
        votes = Counter()
        for fname in os.listdir(folder):
            denom = extract_denomination(fname)
            if denom:
                votes[denom] += 1

        if not votes:
            ambiguous[idx] = "no denomination found in any filename"
            continue

        top_denom, top_count = votes.most_common(1)[0]
        total = sum(votes.values())
        confidence = top_count / total
        if confidence < 0.9:
            ambiguous[idx] = f"inconsistent votes: {dict(votes)}"
        mapping[idx] = top_denom
        print(f"  class_{idx}: {dict(votes)} -> {top_denom} "
              f"({confidence:.0%} confidence)")

    return mapping, ambiguous


def sanity_check(mapping):
    problems = []
    denoms = list(mapping.values())
    missing = set(CANONICAL) - set(denoms)
    if missing:
        problems.append(f"missing denominations: {sorted(missing)}")
    dupes = [d for d, c in Counter(denoms).items() if c > 1]
    if dupes:
        problems.append(f"denominations assigned to more than one class: {dupes}")
    return problems


def patch_merge_script(dataset_name, mapping, script_path="merge_datasets.py"):
    if not os.path.exists(script_path):
        print(f"\n! Could not find {script_path} in current directory — "
              f"skipping auto-patch. Add this manually to "
              f"DATASET_CLASS_INDEX_OVERRIDES instead:\n")
        print_manual_entry(dataset_name, mapping)
        return

    shutil.copy2(script_path, script_path + ".bak")

    with open(script_path, "r") as f:
        content = f.read()

    entry_lines = "\n".join(
        f'        {idx}: "{denom}",' for idx, denom in sorted(mapping.items())
    )
    entry_block = f'    "{dataset_name}": {{\n{entry_lines}\n    }},\n'

    pattern = re.compile(
        r"(DATASET_CLASS_INDEX_OVERRIDES = \{)(.*?)(\n\})", re.DOTALL
    )
    match = pattern.search(content)
    if not match:
        print(f"\n! Could not find DATASET_CLASS_INDEX_OVERRIDES dict in "
              f"{script_path} — add this manually:\n")
        print_manual_entry(dataset_name, mapping)
        return

    new_content = (
        content[: match.start()]
        + match.group(1)
        + match.group(2)
        + "\n"
        + entry_block
        + content[match.end() - 2 :]  # keep the "\n}"
    )

    with open(script_path, "w") as f:
        f.write(new_content)

    print(f"\nPatched {script_path} (backup saved as {script_path}.bak)")


def print_manual_entry(dataset_name, mapping):
    print(f'    "{dataset_name}": {{')
    for idx, denom in sorted(mapping.items()):
        print(f'        {idx}: "{denom}",')
    print("    },")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    parser.add_argument("--crops", default="crops_by_class")
    parser.add_argument("--merge-script", default="merge_datasets.py")
    args = parser.parse_args()

    if not os.path.isdir(args.crops):
        print(f"'{args.crops}' not found — run crop_by_class.py first.")
        return

    dataset_name = os.path.basename(args.dataset_dir.rstrip("/\\"))

    print(f"Scanning '{args.crops}' for denomination hints in filenames...\n")
    mapping, ambiguous = build_mapping(args.crops)

    if ambiguous:
        print("\nSome classes need a manual look (couldn't auto-detect confidently):")
        for idx, reason in ambiguous.items():
            print(f"  class_{idx}: {reason}")
        print("Open a couple of images in that class folder manually to confirm.")

    problems = sanity_check(mapping)
    if problems:
        print("\nSanity check issues:")
        for p in problems:
            print(f"  - {p}")

    print(f"\nFinal mapping for '{dataset_name}':")
    print_manual_entry(dataset_name, mapping)

    if not ambiguous and not problems:
        patch_merge_script(dataset_name, mapping, script_path=args.merge_script)
        print("\nDone. Now re-run: python merge_datasets.py")
    else:
        print("\nFix the issues above first, then re-run this script, or "
              "paste the mapping printed above manually into "
              "DATASET_CLASS_INDEX_OVERRIDES in merge_datasets.py.")


if __name__ == "__main__":
    main()
