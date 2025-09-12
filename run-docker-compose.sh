#!/bin/bash

# Load environment variables from .env safely
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Default to false if not set
USE_GPU=${USE_GPU:-false}

if [ "$USE_GPU" = "true" ]; then
    echo "Running with GPU support."
    docker compose -f docker-compose.gpu.yml up --build
else
    echo "Running without GPU support."
    docker compose -f docker-compose.yml up --build
fi
