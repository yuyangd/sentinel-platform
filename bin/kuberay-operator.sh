#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."

# Add the KubeRay Helm repo
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

# Create dedicated namespace
kubectl create namespace ray-system


# Install the Operator (The Manager)
helm install kuberay-operator kuberay/kuberay-operator --version 1.5.1 --namespace ray-system

# Install metrics server (EKS) This should be install by default in EKS.
helm repo add metrics-server https://kubernetes-sigs.github.io/metrics-server/
helm repo update

helm install metrics-server metrics-server/metrics-server \
  --version 3.13.0 \
  --namespace kube-system

# Verify
kubectl top nodes

# Install the Metrics Server (kind)
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Patch it to work insecurely in Kind
kubectl patch -n kube-system deployment metrics-server --type=json \
  -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'