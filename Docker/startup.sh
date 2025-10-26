#!/bin/sh
PROJECT_ROOT="/workspaces/ML_Project"   
VENV_PATH="$PROJECT_ROOT/.venv"         # Note: env kept in the repo folder

# Install uv globally first
curl -LsSf https://astral.sh/uv/install.sh | sh


make venv

# Ensure every new shell auto-activates the venv
echo 'source /workspaces/ML_Project/.venv/bin/activate' >> /home/vscode/.bashrc
echo 'source /workspaces/ML_Project/.venv/bin/activate' >> /home/vscode/.zshrc

echo 'export PYTHONPATH="/workspaces/ML_Project/llm_utils:${PYTHONPATH}"' >> ~/.profile

# Prepare MLFlow
export MLFLOW_TRACKING_URI="https://.com"
#export MLFLOW_ARTIFACT_URI = "s3://your-s3-bucket/path"