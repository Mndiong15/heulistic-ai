FROM python:3.11-slim
# boto3 so the stub can upload a placeholder model artifact, which makes the
# whole local loop — including `heulistic download` — exercisable without a GPU.
RUN pip install --no-cache-dir requests boto3
COPY train_stub.py /train_stub.py
CMD ["python", "/train_stub.py"]
