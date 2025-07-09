FROM python:3.12-slim

# OCI Image Format Specification labels
LABEL org.opencontainers.image.title="AST CLI"
LABEL org.opencontainers.image.description="AST tools for navigating Python codebases"
LABEL org.opencontainers.image.authors="AST Contributors"
LABEL org.opencontainers.image.source="https://github.com/brukhabtu/ast"
LABEL org.opencontainers.image.documentation="https://github.com/brukhabtu/ast/blob/main/README.md"

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

# Set up non-root user with explicit UID/GID
RUN groupadd -g 1000 astuser && \
    useradd -m -u 1000 -g 1000 -s /bin/bash astuser && \
    chown -R astuser:astuser /app
USER astuser

# Default command
CMD ["python", "-m", "astlib.cli", "--help"]