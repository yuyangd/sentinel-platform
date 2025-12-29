# Install MLFlow

Install a lightweight MLflow server backed by a PVC (Persistent Volume Claim) for artifact storage. In production, this would be S3/RDS

## use custom manifests

k8s/mlflow-minio.yaml to create the s3 bucket

Open UI: http://localhost:9001

k8s/mlflow-platform.yaml to create the postgres and mlflow app


## Check MLflow UI

```bash
# Check the service name
kubectl get svc
# Forward port 5000 (standard MLflow port)
kubectl port-forward service/mlflow 5000:5000
```

Open: http://localhost:5000

