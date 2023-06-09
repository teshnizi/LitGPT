FROM python:3.11-slim

WORKDIR /api

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:80", "--timeout", "180", "api.index:app"]