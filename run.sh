#!/bin/bash

# Build the Docker image
docker build --platform linux/amd64 -t pdf-heading-extractor:latest .

# Run the container
docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" --network none pdf-heading-extractor:latest