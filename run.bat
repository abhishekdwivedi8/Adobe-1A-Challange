@echo off
REM Build the Docker image
docker build --platform linux/amd64 -t pdf-heading-extractor:latest .

REM Run the container
docker run --rm -v "%cd%\input:/app/input" -v "%cd%\output:/app/output" --network none pdf-heading-extractor:latest