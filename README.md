# Car Condition Demo (FastAPI + YOLOv8)

Определение состояния авто по фото: чистый/грязный и битый/целый, с подсветкой дефектов (вмятины, царапины, ржавчина, грязь).

## Быстрый старт (Windows, PowerShell)
- Требования: Python 3.11, Git. GPU не обязателен.
- Клонируйте репозиторий и установите зависимости:

```powershell
git clone https://github.com/Korgan05/car_condition.git
cd car_condition
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

- Запуск веб-сервера (FastAPI) с YOLO и UI:

```powershell
# Вариант 1: CPU/.venv
./run_web.ps1 -YoloWeights "yolov8_runs\rust_scratch_s960_e30_gpu_v2\weights\best.pt;yolov8_runs\car_dent_only_s960_e30_gpu_v2\weights\best.pt"

# Вариант 2: GPU/.venv-gpu (скрипт сам выберет .venv-gpu при наличии)
./run_web.ps1 -YoloWeights "yolov8_runs\rust_scratch_s960_e30_gpu_v2\weights\best.pt;yolov8_runs\car_dent_only_s960_e30_gpu_v2\weights\best.pt"
```

- Откройте: `http://127.0.0.1:8000`
  - Если YOLO не загружается, смотрите `GET /api/metadata` (`yolo_loaded`, `yolo_candidates`, `yolo_error`).

## API: проверка из PowerShell
- Метаданные: модели/классы/пороговые значения

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/metadata | Select-Object -ExpandProperty Content
```

- Предсказание (классификация бинарная):

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8000/api/predict -Method Post -InFile path\to\car.jpg -ContentType 'application/octet-stream' | Select-Object -ExpandProperty Content
```

- Детекция (YOLO), параметр `conf` регулирует уверенность (0.01–0.5):

```powershell
$img='path\to\car.jpg'
Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/detect?conf=0.05" -Method Post -InFile $img -ContentType 'application/octet-stream' | Select-Object -ExpandProperty Content
```

## Где взять веса
- Веса YOLO и датасеты не хранятся в GitHub и игнорируются (`yolov8_runs/`, Roboflow-папки). Поместите свои `best.pt` в `yolov8_runs/*/weights/`.
- Можно указать пути к весам через `-YoloWeights` или положить их в `yolov8_runs/*/weights/{best,last}.pt` для автопоиска.

## Обучение YOLO (опционально)
Установите дополнительные пакеты:

```powershell
pip install -r requirements-train.txt
```

Запуск с ресюмом и ретраями:

```powershell
# Вмятины
.\.venv-gpu\Scripts\python.exe scripts\train_gpu.py --weights yolov8_runs/car_dent_only_s960_e60/weights/best.pt --data "Car_Dent_Only.v1i.yolov8/data.yaml" --project yolov8_runs --name car_dent_only_s960_e30_gpu_v2 --imgsz 960 --epochs 30 --batch 2 --device 0 --workers 2 --resume

# Ржавчина/царапины/вмятины
.\.venv-gpu\Scripts\python.exe scripts\train_gpu.py --weights yolov8_runs/rust_scratch_s960_e140_v3/weights/best.pt --data "Rust and Scrach.v1i.yolov8/data.yaml" --project yolov8_runs --name rust_scratch_s960_e30_gpu_v2 --imgsz 960 --epochs 30 --batch 2 --device 0 --workers 2 --resume
```

## Полезные параметры
- `run_web.ps1`:
  - `-Ckpt` — путь к чекпойнту классификатора (`checkpoints/resnet18_multitask.pt`)
  - `-BindHost` — адрес бинда (по умолчанию `127.0.0.1`)
  - `-Port` — порт (например `8088`)
  - `-Backbone` — `auto` (определится по чекпойнту) или явный resnet18/resnet34
  - `-YoloWeights` — один или несколько путей через запятую к *.pt

- Эндпоинты FastAPI:
  - `/` — статическая страница с UI
  - `/api/metadata` — информация о загруженных моделях
  - `/api/predict` — бинарная классификация (чистый/грязный, целый/повреждённый)
  - `/api/detect?conf=...` — детекция YOLO, `conf` от 0.01 до 0.5
  - `/api/heatmap` и `/api/heatmap_boxes` — тепловые карты/боксы для UI

## Замечания
- Если после изменения кода сервер не стартует, убедитесь, что активирован venv и зависимости установлены без ошибок.
- На CPU инференс работает, но может быть медленнее, чем на GPU.
- В продакшн-режиме используйте отдельный процесс-менеджер и фиксированные версии зависимостей.

## Структура проекта (сжатая)
```
car_condition/
  src/web/main.py           # FastAPI приложение + YOLO и Grad-CAM
  src/web/static/index.html # Веб-UI с отрисовкой боксов и теплокарт
  run_web.ps1               # Скрипт запуска веба на Windows
  yolov8_runs/*/weights/    # Локальные веса YOLO (игнорируются гитом)
  checkpoints/*.pt          # Классификатор (локально)
  requirements.txt          # Зависимости
```

## Лицензии и данные
Соблюдайте лицензии используемых датасетов и библиотек. Удаляйте персональные данные на изображениях (номера/лица) или используйте размытие.
