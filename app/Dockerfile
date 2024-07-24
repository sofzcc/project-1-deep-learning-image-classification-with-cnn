# Use the official TensorFlow Serving image as the base image
FROM tensorflow/serving:latest

# Update the package list and install Python3 and pip3
RUN apt-get update && \
    apt-get install -y python3 python3-pip

# Copy the requirements.txt file and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

# Copy the model to the model directory
COPY model /models/model/1 

# Copy Flask app files
COPY app.py /app/app.py
COPY templates /app/templates

# Copy the startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Set environment variables
ENV MODEL_NAME=model

# Expose the default port TensorFlow Serving uses
EXPOSE 8501

# Expose the Flask app port (if needed)
EXPOSE 5000

# Use CMD to run the startup script
CMD ["sh", "/app/start.sh"]
