# Install Prometheus


## Add helm chart
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
```

## helm install

```bash
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set grafana.enabled=true \
  --set prometheus.service.type=ClusterIP
```

## Tell prometheus to scrape ray

kubectl apply -f k8s/ray-monitor.yaml

## grafana
kubectl port-forward svc/prometheus-grafana 3000:80 -n monitoring

## check docs/k8s_troubleshoot.md
