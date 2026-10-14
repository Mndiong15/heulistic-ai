#!/bin/bash
# Build and push Docker images for all three services to ECR.
# Run from the repo root after terraform apply has created the ECR repos.
#
# Usage:
#   AWS_PROFILE=heulistic-mndiong bash scripts/build-and-push.sh
#
# To redeploy without rebuilding (e.g. just restart the containers on EC2):
#   bash scripts/deploy.sh   (see inline comments below)
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
PREFIX="ftp-prod"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

ECR_BASE="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"

echo "Logging in to ECR..."
aws ecr get-login-password --region "$REGION" | \
  docker login --username AWS --password-stdin "$ECR_BASE"

for SERVICE in api scheduler training; do
  REPO="$ECR_BASE/$PREFIX/$SERVICE"
  echo ""
  echo "Building $SERVICE → $REPO:latest"
  docker build -t "$REPO:latest" "$ROOT/$SERVICE"

  echo "Pushing $SERVICE → $REPO:latest"
  docker push "$REPO:latest"
  echo "$SERVICE pushed."
done

echo ""
echo "All images pushed. To redeploy running instances, use SSM Run Command:"
echo ""
echo "  # API"
echo "  aws ssm send-command \\"
echo "    --region $REGION \\"
echo "    --targets 'Key=tag:Name,Values=$PREFIX-api' \\"
echo "    --document-name 'AWS-RunShellScript' \\"
echo "    --parameters 'commands=[\"docker pull \$(aws ssm get-parameter --name /$PREFIX/ecr/api-url --region $REGION --query Parameter.Value --output text):latest && docker restart api\"]'"
echo ""
echo "  # Scheduler"
echo "  aws ssm send-command \\"
echo "    --region $REGION \\"
echo "    --targets 'Key=tag:Name,Values=$PREFIX-scheduler' \\"
echo "    --document-name 'AWS-RunShellScript' \\"
echo "    --parameters 'commands=[\"docker pull \$(aws ssm get-parameter --name /$PREFIX/ecr/scheduler-url --region $REGION --query Parameter.Value --output text):latest && docker restart scheduler\"]'"
