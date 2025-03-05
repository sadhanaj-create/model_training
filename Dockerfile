# Base Image
#FROM pytorch/pytorch:latest

# Use an ARM64-compatible base image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files
COPY . /app


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN python data_preprocessing.py
RUN python model.py

# Set the Python path so that the test files can locate the application modules
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8080

# Run the tests
RUN pytest tests/
# Run tests with some options to limit output and failures
#RUN pytests --maxfail=5 --disable-warnings -q

# Run the application
CMD ["python3", "app.py"]
