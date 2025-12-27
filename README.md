# Sentinel Platform

Sentinel is a reference implementation of a modern **AI Infrastructure Platform**.

It demonstrates how to provision and manage a distributed compute environment for Machine Learning (using **Ray**) on top of Kubernetes, following **Infrastructure-as-Code** (IaC) principles with Terraform.

## Tech Stack
* **Orchestration:** Kubernetes (Kind)
* **Compute Framework:** KubeRay (Ray Cluster)
* **IaC:** Terraform
* **Scripting:** Python

## Open Ray Dashboard

```bash
kubectl port-forward service/sentinel-ray-cluster-head-svc 8265:8265
```

Browser: http://localhost:8265/

## try inference

```bash
kubectl port-forward service/sentinel-service-head-svc 8000:8000
```

Curl

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"text": "Ray Serve is amazing!"}'
```
