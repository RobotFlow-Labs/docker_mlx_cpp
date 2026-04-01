FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir openai==1.58.1

COPY app.py .

CMD ["python", "app.py"]
