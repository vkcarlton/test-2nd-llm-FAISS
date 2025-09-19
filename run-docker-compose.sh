#!/bin/bash

# Load environment variables from .env safely
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Default to false if not set
USE_GPU=${USE_GPU:-false}
LOCAL=${LOCAL:-true}

if [ "$LOCAL" = "true" ]; then
    echo "Running locally"
    if [ "$USE_GPU" = "true" ]; then
            echo "Running with GPU support."
            docker compose -f ./compose/docker-compose.local.gpu.yml up --build
        else
            echo "Running without GPU support."
            docker compose -f ./compose/docker-compose.local.yml up --build
        fi
else
    echo "Running with Cloudflare Tunnel Open"

    if [ "$USE_GPU" = "true" ]; then
        echo "Running with GPU support."
        docker compose -f ./compose/docker-compose.gpu.yml up --build
    else
        echo "Running without GPU support."
        docker compose -f ./compose/docker-compose.yml up --build
    fi
fi

