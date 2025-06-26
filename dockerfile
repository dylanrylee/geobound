# Use a slim Python base (switch to pytorch/cuda image if GPU support needed)
FROM python:3.9-slim

# Install GDAL and OpenCV system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev \
    build-essential \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# Point GDAL headers at the right place
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Set working dir to your project
WORKDIR /geobound

# Copy and install Python dependencies
COPY requirements.txt /geobound/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . /geobound/

# Default command: adjust to your training entrypoint
CMD ["python", "train.py"]