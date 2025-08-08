# Use official Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .

# Install PyTorch separately because of special URL
RUN pip install -r requirements.txt
RUN pip install torch==2.8.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Copy app code
COPY . .

# Expose port
EXPOSE 8000

# Run the app with uvicorn
CMD ["uvicorn", "server.api:app", "--host", "0.0.0.0", "--port", "8000"]
