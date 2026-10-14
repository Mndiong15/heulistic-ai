FROM python:3.11-slim

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Shared Axolotl schema and config validation, used by both the API and the
# CLI. Installed from the monorepo until it is published to PyPI — which is
# why this image builds from the repo root rather than ./api.
COPY packages/heulistic-axolotl /tmp/heulistic-axolotl
RUN pip install --no-cache-dir /tmp/heulistic-axolotl && rm -rf /tmp/heulistic-axolotl

COPY api/ .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
