# Base image
FROM python:3.8-slim

# Install necessary packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && apt-get clean

# Install Flask and other dependencies
COPY Requirements.txt /app/Requirements.txt
RUN pip3 install -r /app/Requirements.txt

# Copy Flask app files and templates
COPY app.py /app/app.py
COPY templates /app/templates
COPY uploads /app/uploads

# Set the working directory
WORKDIR /app

# Expose Flask port
EXPOSE 5000

# Command to run the Flask app
CMD ["python3", "app.py"]
