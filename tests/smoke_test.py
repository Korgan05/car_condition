import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))

# 1) генерация синтетики
subprocess.check_call([sys.executable, os.path.join(ROOT, 'scripts', 'create_dummy_dataset.py')])

# 2) сплиты
subprocess.check_call([sys.executable, os.path.join(ROOT, 'scripts', 'prepare_split.py'), '--labels', os.path.join(ROOT, 'data', 'labels.csv'), '--val-size', '0.2'])

print('Smoke test: dataset prepared. You can now run training:')
print('  python -m src.train --train-csv data/splits/train.csv --val-csv data/splits/val.csv --epochs 1 --batch-size 8')
