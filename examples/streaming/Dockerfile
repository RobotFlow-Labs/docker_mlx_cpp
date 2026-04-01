FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir httpx rich
COPY stream_chat.py .
CMD ["python", "stream_chat.py"]
