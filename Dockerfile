# Use the official Python image as a base image
FROM python:3.12-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT 8080

# Command to run the application using Gunicorn for production-ready Flask deployment
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT}"]
