# STAGE 1: Builder
# Updated to Python 3.13 as requested
FROM python:3.13-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# STAGE 2: Runtime
FROM python:3.13-slim

WORKDIR /app

# Copy installed packages from Builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy source code
COPY src/ ./src/

# Create assets folder
RUN mkdir -p assets

# Expose the port
EXPOSE 8000

# Command to run the server
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]