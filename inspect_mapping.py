from pathlib import Path
from collections import Counter, defaultdict

path = Path('data/raw/INDIAN CURRENCY DETECTION.v4i.yolov8')
tokens = ['100IndianRupees', '50IndianRupees', '20IndianRupees', '10IndianRupees', '5IndianRupees', '200IndianRupees', '500IndianRupees', '2000IndianRupees']

def denom_from_name(name):
    for token in tokens:
        if token in name:
            return token
    return None

for split in ['train', 'valid', 'test']:
    p = path / split / 'labels'
    counts = Counter()
    mapping = defaultdict(Counter)
    for f in sorted(p.glob('*.txt')):
        text = f.read_text().strip()
        if not text:
            continue
        cls = int(text.split()[0])
        denom = denom_from_name(f.name)
        counts[cls] += 1
        mapping[cls][denom] += 1
    print(f'=== {split} ===')
    print('counts', dict(sorted(counts.items())))
    for cls, counter in sorted(mapping.items()):
        print(cls, dict(counter))
    print()
