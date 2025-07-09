FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY astlib ./astlib

# Install the package
RUN pip install --no-cache-dir -e .

# Set up non-root user
RUN useradd -m -s /bin/bash astuser && chown -R astuser:astuser /app
USER astuser

# Default command
CMD ["python", "-m", "astlib.cli", "--help"]