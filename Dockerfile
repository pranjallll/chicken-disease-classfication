FROM python:3.9-slim

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port 8080 for Azure
EXPOSE 8080

# Use Gunicorn (production WSGI server) instead of python app.py
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]

