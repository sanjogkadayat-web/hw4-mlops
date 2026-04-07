FROM python:3.10-slim

WORKDIR /app

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY . .

EXPOSE 5000

# Use gunicorn, NOT Flask's dev server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
