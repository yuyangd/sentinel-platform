# Ping

```bash
kubectl exec -it sentinel-training-cluster-head-xxxx -n sentinel-prod -- curl -I http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090/graph
```

## access prometheus

```bash
kubectl port-forward svc/prometheus-kube-prometheus-prometheus 9090:9090 -n monitoring

# http://localhost:9090/targets


# get Grafana password
kubectl get secret prometheus-grafana -n monitoring -o jsonpath="{.data.admin-password}" | base64 --decode ; echo


# set configmaps for grafana

kubectl get configmaps -n monitoring

# edit
kubectl edit configmap prometheus-grafana -n monitoring

# Add the following:
# [security]
# allow_embedding = true
# cookie_samesite = none
# cookie_secure = true

# restart grafana
# Verify the deployment name first (it is usually prometheus-grafana)
kubectl get deployments -n monitoring | grep grafana

# Restart it (assuming the name is prometheus-grafana)
kubectl rollout restart deployment prometheus-grafana -n monitoring

# To get the Ray dashboard in Grafana, we must import the Ray Dashboard json

# 1. Download the Default Dashboard (System metrics)
kubectl -n sentinel-prod cp $HEAD_POD:/tmp/ray/session_latest/metrics/grafana/dashboards/default_grafana_dashboard.json ./ray-dashboards/default_grafana_dashboard.json

# 2. Download the Serve Dashboard (Serve overview)
kubectl -n sentinel-prod cp $HEAD_POD:/tmp/ray/session_latest/metrics/grafana/dashboards/serve_grafana_dashboard.json ./ray-dashboards/serve_grafana_dashboard.json

# 3. Download the Serve Deployment Dashboard (Per-model metrics)
kubectl -n sentinel-prod cp $HEAD_POD:/tmp/ray/session_latest/metrics/grafana/dashboards/serve_deployment_grafana_dashboard.json ./ray-dashboards/serve_deployment_grafana_dashboard.json

# ClickOps import into Grafana via UI
```

## Check EKS kubernetes version

```bash
aws eks describe-cluster --name du-yuyang-training --query 'cluster.version' --output text
```