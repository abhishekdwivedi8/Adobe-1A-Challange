# Docker Instructions

## Building the Docker Image

```bash
docker build --platform linux/amd64 -t pdf-heading-extractor:latest .
```

## Running the Container

```bash
docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" --network none pdf-heading-extractor:latest
```

## Requirements

- CPU architecture: amd64 (x86_64)
- No GPU dependencies
- Model size: < 200MB (current model is ~250KB)
- Works offline - no network/internet calls

## Performance

- Execution time: ≤ 10 seconds for a 50-page PDF
- Optimized for systems with 8 CPUs and 16 GB RAM

## Docker Environment

The Docker container is configured to:
1. Automatically process all PDFs from `/app/input` directory
2. Generate a corresponding JSON file in `/app/output` for each PDF
3. Run without internet access
4. Utilize CPU resources efficiently

## Troubleshooting

If you encounter any issues:

1. Ensure input and output directories exist and have proper permissions
2. Check that the model file is present in the model directory
3. Verify that the PDFs are valid and readable