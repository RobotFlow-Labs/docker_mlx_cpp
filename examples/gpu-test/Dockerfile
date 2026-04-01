FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir httpx rich

COPY test_all_gpu_ops.py .

CMD ["python", "test_all_gpu_ops.py"]
