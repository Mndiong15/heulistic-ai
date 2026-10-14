import os
import sys

# Add training root to path so tests can import train
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Minimal env vars required at train.py module level before import
os.environ.setdefault("JOB_ID", "test-job-00000000")
os.environ.setdefault("API_URL", "http://localhost:8000")
os.environ.setdefault("S3_BUCKET", "test-bucket")
os.environ.setdefault("CONFIG_S3_PATH", "s3://test-bucket/configs/test/config.yaml")
os.environ.setdefault("INSTANCE_ID", "i-testinstance")
os.environ.setdefault("AWS_REGION", "us-east-1")
