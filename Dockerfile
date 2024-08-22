# Use an official Python image as the base
FROM python:3.9-slim

ENV DEBUG_INFERENCE=true
ENV DEBUG_DIGIT_IMAGES=false

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install asgiref
RUN pip install uvicorn

# Copy the application code
COPY . /app/

# Expose the port
EXPOSE 8000

# Run the command to start the development server
CMD ["uvicorn", "main:wsgi", "--host", "0.0.0.0", "--port", "8000"]