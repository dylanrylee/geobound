# Use official Python slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (e.g. GDAL for geospatial libs)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libgdal-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy all your code into the container
COPY . .

# Default command to run your classifier
CMD ["python", "land-classifier.py"]
