FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ /app/app/
COPY model/ /app/model/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create directories
RUN mkdir -p /app/input /app/output /tmp/pdf_processing

# Run the application
CMD ["python", "-m", "app.main"]