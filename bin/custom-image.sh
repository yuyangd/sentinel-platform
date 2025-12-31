#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."


docker build -f docker/ray/Dockerfile.serve --progress=plain -t duyuyang545/sentinel-serve:v1 .
docker push duyuyang545/sentinel-serve:v1

# kubectl apply -f k8s/ray-service.yaml

docker build -f docker/ray/Dockerfile.train --progress=plain -t duyuyang545/sentinel-train:v1 .
docker push duyuyang545/sentinel-train:v1

# kubectl apply -f k8s/ray-cluster-train.yaml
