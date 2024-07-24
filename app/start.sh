#!/bin/bash

# Start Flask app in the background
python3 /app.py &

# Start TensorFlow Serving
tensorflow_model_server --rest_api_port=8501 --model_name=${MODEL_NAME} --model_base_path=/models/model
