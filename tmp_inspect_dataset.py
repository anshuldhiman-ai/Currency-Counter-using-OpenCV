from pathlib import Path
from collections import Counter

path = Path('data/raw/INDIAN CURRENCY DETECTION.v4i.yolov8')
for split in ['train', 'valid', 'test']:
    p = path / split / 'labels'
    counts = Counter()
    files = sorted(p.glob('*.txt'))
    for f in files:
        text = f.read_text().strip()
        if not text:
            continue
        cls = int(text.split()[0])
        counts[cls] += 1
    print(split, 'files', len(files), 'counts', dict(sorted(counts.items())))
    print('sample names:', [f.name for f in files[:5]])
