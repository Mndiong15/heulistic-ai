FROM python:3.11-slim

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The heulistic package, for heulistic.axolotl — the Axolotl config schema,
# validator and GPU planner the CLI also uses. Installed from this checkout so
# the API always matches it, which is why the image builds from the repo root.
COPY cli /tmp/heulistic
RUN pip install --no-cache-dir /tmp/heulistic && rm -rf /tmp/heulistic

COPY api/ .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
