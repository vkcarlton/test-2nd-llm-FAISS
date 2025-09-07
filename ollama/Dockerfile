FROM ollama/ollama:latest

RUN ollama serve
# Pre-pull llama3 when the container is built
CMD ["pull", "llama3"]
