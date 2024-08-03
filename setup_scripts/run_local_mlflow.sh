#!/bin/bash

# Start the MLflow UI in the background
nohup poetry run mlflow ui &
ui_pid=$!  # Capture the process ID of the background process
echo ui_pid: $ui_pid

# Run your main script
poetry run python -m src.main

# Wait for the main script to finish
# wait

# Terminate the MLflow UI process using its PID
kill $ui_pid

echo "MLflow UI stopped."
