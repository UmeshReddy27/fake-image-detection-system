# Use an official, lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the server
WORKDIR /app

# IMPORTANT: OpenCV (cv2) requires these system libraries to process images!
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the server
COPY . .

# Hugging Face Spaces REQUIRES apps to run on port 7860
ENV PORT=7860
EXPOSE 7860

# Boot up the heavy-duty Gunicorn server
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]