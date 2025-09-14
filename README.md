# Car Condition: Веб-приложение с детекцией

Определение состояния авто по фото с классификацией (чистый/грязный, целый/повреждённый) и подсветкой проблем на изображении (YOLO: царапина, вмятина, ржавчина, грязь).

## Быстрый старт (Windows, PowerShell)
- Требования: Python 3.10–3.11, Git. GPU не обязателен.
- Клонируйте репозиторий и установите зависимости в виртуальное окружение:

```powershell
git clone https://github.com/Korgan05/car_condition.git
cd car_condition
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

- Запуск веб-сервера (FastAPI) с детекцией YOLO и UI:

```powershell
# вариант 1: один вес (ржавчина/царапины)
./run_web.ps1 -Ckpt checkpoints/resnet18_multitask.pt -BindHost 127.0.0.1 -Port 8088 -YoloWeights 'yolov8_runs\rust_scratch\weights\best.pt'

# вариант 2: ансамбль двух весов (ржавчина+царапины, царапины+вмятины)
./run_web.ps1 -Ckpt checkpoints/resnet18_multitask.pt -BindHost 127.0.0.1 -Port 8088 -YoloWeights 'yolov8_runs\rust_scratch\weights\best.pt,yolov8_runs\car_scratch_dent\weights\best.pt'
```

- Откройте в браузере: `http://127.0.0.1:8088`
  - Кнопка для загрузки фото; ползунок чувствительности (сдвиньте вниз до 0.01 для максимальной полноты по ржавчине).
  - Цветные рамки: красный — ржавчина, синий — царапина, оранжевый — вмятина, коричневый — грязь.
  - Если YOLO ничего не находит, включается тепловая карта/цветовой фолбэк.

## Проверка API из PowerShell
- Метаданные: модели/классы/пороговые значения

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8088/api/metadata | Select-Object -ExpandProperty Content
```

- Предсказание (классификация бинарная):

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8088/api/predict -Method Post -InFile path\to\car.jpg -ContentType 'application/octet-stream' | Select-Object -ExpandProperty Content
```

- Детекция (YOLO), параметр `conf` регулирует уверенность (0.01–0.5):

```powershell
$img='path\to\car.jpg'
Invoke-WebRequest -Uri "http://127.0.0.1:8088/api/detect?conf=0.05" -Method Post -InFile $img -ContentType 'application/octet-stream' | Select-Object -ExpandProperty Content
```

## Где взять веса
- В репозитории уже есть обученные веса YOLO (папка `yolov8_runs/*/weights/best.pt`). Этого достаточно для запуска.
- Для дообучения/переобучения скачайте датасеты на Roboflow:
  - Rust and Scrach.v1i.yolov8
  - Car Scratch and Dent.v1i.yolov8
  Сохраните в корень проекта одноимёнными папками (как в названии).

## Обучение YOLO (опционально)
Запустить обучение на CPU (для быстрого прогресса, рекомендуется больше эпох и GPU при наличии):

```powershell
Set-Location .\car_condition
$env:ULTRALYTICS_SETTINGS = "$((Get-Location).Path)\ultralytics_settings.yaml"
.\.venv\Scripts\python.exe -m scripts.train_yolo --data-yaml 'Car Scratch and Dent.v1i.yolov8/data.yaml' --model yolov8n.pt --imgsz 640 --epochs 10 --batch 4 --device cpu
```

Или напрямую через ultralytics:

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; m=YOLO('yolov8n.pt'); m.train(data='Rust and Scrach.v1i.yolov8/data.yaml', imgsz=640, epochs=10, batch=4, device='cpu', project='yolov8_runs', name='rust_scratch', pretrained=True, exist_ok=True)"
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
  yolov8_runs/*/weights/    # Готовые веса YOLO (best.pt, last.pt)
  checkpoints/*.pt          # Классификатор (мультитаск)
  requirements.txt          # Зависимости
```

## Лицензии и данные
Соблюдайте лицензии используемых датасетов и библиотек. Удаляйте персональные данные на изображениях (номера/лица) или используйте размытие.
