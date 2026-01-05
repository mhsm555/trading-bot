# 1. Base Image
FROM python:3.10-slim

# 2. Set Working Directory
WORKDIR /app

# 3. Copy Dependencies first (Caching layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Code
COPY src/ ./src/
COPY data/ ./data/
# Note: In production, we usually mount data/ as a volume, but this works for a demo.

# 5. Expose Port
EXPOSE 8000

# 6. Run Server
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]