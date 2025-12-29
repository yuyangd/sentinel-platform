#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."


docker build -f docker/ray/Dockerfile --progress=plain -t duyuyang545/sentinel:v3 .
docker push duyuyang545/sentinel:v3

# kubectl apply -f ray-service.yaml