# Use a minimal Python base image
FROM python:3.8-slim

WORKDIR /app
<<<<<<< HEAD

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

=======
>>>>>>> cf9e4a5bc88fc09caf3be65a035dda16d737e846
COPY requirements.txt .
RUN pip --default-timeout=100 install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=app.py

EXPOSE 5001

CMD ["flask", "run", "--host=0.0.0.0"]