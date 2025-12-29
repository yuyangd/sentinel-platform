#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."


docker build -t duyuyang545/sentinel:v1 .
docker push duyuyang545/sentinel:v1

# kubectl apply -f ray-service.yaml