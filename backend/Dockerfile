FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .

# Install a smaller CPU-only torch build
RUN pip install torch torchvision
    
RUN pip install -r requirements.txt

# Copy code and data
COPY . .

EXPOSE 5000

# Run Flask
CMD ["python", "app.py"]
