# 1. Use an official lightweight Python base image
FROM python:3.10-slim

# 2. Set environment variables to prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Copy requirements.txt first to take advantage of Docker layer caching
COPY requirements.txt .

# 5. Install dependencies inside the container
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your application code and trained model binaries
COPY . /app

# 7. Expose port 8000 for FastAPI
EXPOSE 8000

# 8. Command to run the FastAPI app via Uvicorn (0.0.0.0 is required inside Docker)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]