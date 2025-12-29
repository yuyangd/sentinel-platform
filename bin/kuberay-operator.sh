#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."

# Add the KubeRay Helm repo
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

# Install the Operator (The Manager)
helm install kuberay-operator kuberay/kuberay-operator --version 1.5.1

# Install the Metrics Server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Patch it to work insecurely in Kind
kubectl patch -n kube-system deployment metrics-server --type=json \
  -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'