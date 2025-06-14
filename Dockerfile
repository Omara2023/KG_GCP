# Use the official Python image as a base image
FROM python:3.12-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Define the port that the container will listen on
ENV PORT 8080

# Command to run the application using Gunicorn for production-ready Flask deployment
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app"]