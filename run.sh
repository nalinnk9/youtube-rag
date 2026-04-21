#!/bin/bash
set -e

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example."
  echo "Open it, add your API keys, and re-run this script."
  exit 1
fi

uvicorn backend.main:app --reload --port 8000
