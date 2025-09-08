#!/bin/bash

# Start JupyterLab for Paperspace Gradient
echo "Starting JupyterLab on port 8888..."
echo "Access your notebook at: http://localhost:8888"

# Start JupyterLab with proper configuration for Paperspace
exec jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --NotebookApp.disable_check_xsrf=True
