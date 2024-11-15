FROM python:3.10.14-bullseye

# Install necessary dependencies
RUN apt-get update && \
  apt-get install -y \
  locales \
  locales-all && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Set locale environment variables
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN groupadd -g 900 mesop && useradd -u 900 -s /bin/bash -g mesop mesop
USER mesop

# Copy application code
COPY . /srv/gender-equity-navigator-app
WORKDIR /srv/gender-equity-navigator-app

# Run Mesop through mesop command
#CMD ["mesop", "main.py"]

# Run Mesop through gunicorn. Should be available at localhost:8080
CMD ["gunicorn", "--worker-class", "gevent", "--workers", "1", "--bind", "0.0.0.0:8080", "main:me", "--log-level=debug"]

# gcloud auth login
# gcloud config set project gender-equity-navigator
# gcloud services enable containerregistry.googleapis.com
# code ~/.docker/config.json
# gcloud auth configure-docker


#Build the Multi-Platform Image
#docker buildx create --use
#docker buildx build --platform linux/amd64,linux/arm64 -t gcr.io/gender-equity-navigator/gender-equity-navigator-app:latest --push .

#docker run -d -p 8080:8080 gcr.io/gender-equity-navigator/gender-equity-navigator-app:latest

#docker build -t gender-equity-navigator-app:latest .
#docker run -d -p 8080:8080 gender-equity-navigator-app:latest 
