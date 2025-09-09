FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .

# Installs cpu only lighter version of torch
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    
RUN pip install -r requirements.txt

# Copy code and data
COPY . .

EXPOSE 5000

# Run Flask
CMD ["python", "app.py"]
