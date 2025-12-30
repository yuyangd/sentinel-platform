# Sentinel Platform

Sentinel is a reference implementation of a modern **AI Infrastructure Platform**.

It demonstrates how to provision and manage a distributed compute environment for Machine Learning (using **Ray**) on top of Kubernetes, following **Infrastructure-as-Code** (IaC) principles with Terraform.

## Tech Stack
* **Orchestration:** Kubernetes (Kind)
* **Compute Framework:** KubeRay (Ray Cluster)
* **IaC:** Terraform
* **Scripting:** Python

## Create kubernetes cluster

```bash
kind create cluster --config kind-config.yaml --name sentinel-platform
```

## Install kuberay

bin/install-ray.sh

## create sentinel service

bin/create-service.sh

## Open Ray Dashboard

```bash
kubectl port-forward svc/sentinel-service-head-svc 8265:8265
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

## Submit training job

To upload the local train/train_job.py to the cluster, use `--working-dir`, otherwise, ray assume the file exists

```bash
ray job submit \
  --working-dir . \
  --runtime-env-json='{"pip": ["transformers", "datasets", "evaluate", "scikit-learn", "mlflow", "torch", "psycopg2-binary", "accelerate"], "env_vars": {"MLFLOW_TRACKING_URI": "http://mlflow:5000"}}' \
  -- python train/train_job.py
```

## Install ekscli


homebrew

```
brew tap aws/tap
brew install aws/tap/eksctl
```

Create an EKS cluster

```
eksctl create cluster -f eks/cluster.yaml --with-oidc
```

create a namespace

```
kubectl create namespace sentinel-prod

kubectl apply -f k8s/ray-service.yaml
```

Tag the subnets, so eks can place the loadbalancer

```
--tags Key=kubernetes.io/role/elb,Value=1
``

## Serve

```
curl -X POST http://ad5a958df008b47d9b082f731650947a-717393962.ap-southeast-2.elb.amazonaws.com:8000/check \
  -H "Content-Type: application/json" \
  -d '{
    "query": "This property has 2 bedrooms and a pool.",
    "facts": {
        "bedrooms": 2,
        "pool": true
    }
  }'
```

## Clean the eks cluster

```
kubectl delete svc sentinel-service-serve-svc -n sentinel-prod
eksctl delete cluster -f eks/cluster.yaml
```
