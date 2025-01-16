FROM python:3.12-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

COPY . /app

# Create all needed directories
RUN mkdir -p /app/runs \
    && mkdir -p /app/models/ddpg \
    && chown -R 1000:1000 /app \
    && chmod -R 777 /app/hyperparameters \
    && chmod -R 777 /app/runs

# Switch to non-root
USER 1000

CMD ["python", "main.py"]
