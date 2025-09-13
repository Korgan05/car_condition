# Car Condition Classifier

Классификация состояния автомобиля по фото: чистота (clean/dirty) и целостность (damaged/intact).

## Возможности
- Двухголовая модель (мультиметка): clean vs dirty и damaged vs intact
- Обучение на своих данных (CSV: filepath, clean, damaged)
- Инференс по одному изображению и батчем
- Демо UI (Gradio) для загрузки фото
- Синтетический датасет для быстрых проверок

## Структура
```
car_condition/
  src/
    data.py          # датасеты/трансформации/сплиты
    model.py         # модель ResNet18 + две головы
    train.py         # обучение/валидация/сохранение чекпойнта
    predict.py       # инференс по изображению/папке
    app.py           # демо UI на Gradio
  scripts/
    prepare_split.py       # формирование train/val из CSV с путями
    create_dummy_dataset.py# синтетика для smoke-тестов
  data/
    raw/            # исходные данные (из Roboflow и др.)
    labels.csv      # метки формата: filepath,clean,damaged
    splits/         # train.csv, val.csv
  checkpoints/      # сохранённые модели
  tests/
    smoke_test.py   # быстрый прогон на синтетике
  requirements.txt
  README.md
  .gitignore
```

## Установка
1) Создать и активировать виртуальное окружение (опционально).
2) Установить зависимости:
```
pip install -r requirements.txt
```

PyTorch: выберите подходящую сборку под вашу систему/GPU с https://pytorch.org/get-started/locally/ при необходимости.

## Данные
- Требуемый формат CSV: `data/labels.csv` с колонками:
  - `filepath` — путь до изображения (от корня проекта или абсолютный)
  - `clean` — 1 если чистый, 0 если грязный
  - `damaged` — 1 если битый/повреждённый, 0 если целый
- Разделение на сплиты:
```
python -m scripts.prepare_split --labels data/labels.csv --val-size 0.2
```
Создаст `data/splits/train.csv` и `data/splits/val.csv`.

Источники данных (пример):
- https://universe.roboflow.com/seva-at1qy/rust-and-scrach
- https://universe.roboflow.com/carpro/car-scratch-and-dent
- https://universe.roboflow.com/project-kmnth/car-scratch-xgxzs

Примечание: соблюдайте лицензионные условия датасетов и избегайте сохранения госномеров.

## Обучение
```
python -m src.train \
  --train-csv data/splits/train.csv \
  --val-csv data/splits/val.csv \
  --epochs 10 \
  --batch-size 16 \
  --lr 3e-4 \
  --out checkpoints/resnet18_multitask.pt
```
Аргументы смотрите `python -m src.train -h`.

## Инференс
```
python -m src.predict --ckpt checkpoints/resnet18_multitask.pt --image path/to/car.jpg
```
Или на папке:
```
python -m src.predict --ckpt checkpoints/resnet18_multitask.pt --folder path/to/images
```

## Демо UI
```
python -m src.app --ckpt checkpoints/resnet18_multitask.pt --server-port 7860
```
Откроется интерфейс в браузере.

## Веб-сайт (FastAPI)
Поднимите сайт с формой загрузки и REST API:
```
python -m src.web.main --ckpt checkpoints/resnet18_multitask.pt --host 127.0.0.1 --port 8000
```
Откройте: http://127.0.0.1:8000 — статическая страница отправляет файл на /api/predict.

Пример REST-запроса (PowerShell):
```
# Отправка изображения
Invoke-WebRequest -Uri http://127.0.0.1:8000/api/predict -Method Post -InFile path\to\car.jpg -ContentType 'application/octet-stream'
```

## Метрики
- На валидации считаются accuracy, precision, recall, f1 для каждой головы, а также среднее.

### Оценка и матрицы ошибок
```
python scripts/evaluate.py --val-csv data/splits/val.csv --ckpt checkpoints/resnet18_multitask.pt --out-dir eval
```
Сохранит `eval/cm_clean.png` и `eval/cm_damaged.png`.

## Ограничения и этика
- Конфиденциальность: избегайте распознаваемых лиц/номеров; храните данные безопасно.
- Bias: разные камеры/условия освещения могут влиять; используйте аугментации и баланс классов.
- Не используйте модель вне домена данных без дополнительной адаптации.

## План улучшений
- Больше данных и таргетированные аугментации (снег/дождь/пыль)
- Многоклассовая градация грязи/повреждений
- Детекция/сегментация для локализации повреждений
- Калибровка вероятностей и активное обучение

## Конвертер Roboflow → labels.csv
Ожидаем структуру папок: `root/clean`, `root/dirty`, `root/damaged`, `root/intact`.
```
python scripts/roboflow_to_labels.py --root path/to/root --out data/labels.csv
python scripts/prepare_split.py --labels data/labels.csv --val-size 0.2
```

## Batch-инференс на папке
```
python scripts/batch_predict.py --ckpt checkpoints/resnet18_multitask.pt --folder path/to/folder --out predictions.csv
```

## Docker
Собрать образ и запустить сайт:
```
docker build -t car-condition .
docker run --rm -p 8000:8000 -e CKPT=checkpoints/resnet18_multitask.pt car-condition
```

## CI
Добавлен GitHub Actions `ci.yml`: установка зависимостей, синтетический датасет, обучение 1 эпоху, предсказание одного изображения.
