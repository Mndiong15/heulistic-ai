FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:22-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# NEXT_PUBLIC_LOCAL_MODE is always true for this image.
# API_URL is intentionally NOT baked in — set at runtime via docker-compose.
ENV NEXT_PUBLIC_LOCAL_MODE=true \
    NEXT_OUTPUT=standalone \
    NEXT_TELEMETRY_DISABLED=1

RUN npm run build

# Guarantee public/ exists. This app ships no static assets, and git cannot
# store an empty directory — so a fresh checkout has no public/ and the COPY
# below fails, even though a local build succeeds where the empty directory
# happens to exist on disk.
RUN mkdir -p /app/public

FROM node:22-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000
CMD ["node", "server.js"]
