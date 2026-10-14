import os
import sys

# Add scheduler root to path so tests can import provisioner, watchdog, config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Minimal env vars required by pydantic-settings before any app module is imported
os.environ.setdefault("DATABASE_URL", "postgresql://dummy/dummy")
os.environ.setdefault("API_INTERNAL_URL", "http://localhost:8000")
os.environ.setdefault("PROJECT_PREFIX", "ftp-test")
os.environ.setdefault("AWS_REGION", "us-east-1")
