# Use Python 3.7 slim-buster as the base image
FROM python:3.7-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy our installation script
COPY docker-scripts/install-deps.sh /tmp/
COPY docker-scripts/run-test.sh /tmp/

# Make scripts executable
RUN chmod +x /tmp/install-deps.sh /tmp/run-test.sh

# Run the installation script
RUN /tmp/install-deps.sh

# Set environment variables
ENV PYTHONPATH="/app:/posformer:${PYTHONPATH}"

# Default command
CMD ["/bin/bash"]