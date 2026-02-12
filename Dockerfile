FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY server.py index.html ./

EXPOSE 10000
CMD ["gunicorn", "server:app", "--bind", "0.0.0.0:10000", "--timeout", "300", "--workers", "2"]
