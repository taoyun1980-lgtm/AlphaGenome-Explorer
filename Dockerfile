FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY server.py index.html ./

EXPOSE 7860
CMD ["gunicorn", "server:app", "--bind", "0.0.0.0:7860", "--timeout", "300", "--workers", "2"]
