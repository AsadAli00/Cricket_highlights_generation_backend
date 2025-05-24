# Use a minimal Python base image
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip --default-timeout=100 install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=app.py

EXPOSE 5001

CMD ["flask", "run", "--host=0.0.0.0"]