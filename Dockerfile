# Простой образ для сайта (FastAPI)
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Отключаем Gradio/тренировки, запускаем только веб
ENV PYTHONUNBUFFERED=1

# Значения по умолчанию
ENV CKPT=checkpoints/resnet18_multitask.pt
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000
CMD ["python", "-m", "src.web.main", "--ckpt", "${CKPT}", "--host", "${HOST}", "--port", "${PORT}"]
