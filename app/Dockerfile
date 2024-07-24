# Use the official TensorFlow Serving image as the base image
FROM tensorflow/serving:latest

# Install Python3 and pip3 if not already present
RUN apt-get update && \
    apt-get install -y python3 python3-pip

# Copy the requirements.txt file and install Python dependencies
COPY app/requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt

# Copy the model to the model directory
COPY app/model /models/model

# Copy Flask app files
COPY app/app.py /app.py
COPY app/templates /templates
COPY app/static /static
COPY app/uploads /uploads

# Copy the startup script
COPY /start.sh /start.sh
RUN chmod +x /start.sh

# Set environment variables
ENV MODEL_NAME=model

# Expose the default port TensorFlow Serving uses
EXPOSE 8501

# Expose the Flask app port (if needed)
EXPOSE 5000

# Use CMD to run the startup script
CMD ["sh", "/start.sh"]
