# Sentinel Platform Architecture

Sentinel is an end-to-end **AI Infrastructure Platform** demonstrating distributed machine learning on Kubernetes. This document outlines the system components, data flow, and deployment topology.

---

## System Overview

```mermaid
graph TB
    subgraph AWS["AWS Cloud"]
        EKS["Amazon EKS<br/>(Kubernetes Cluster)"]
        S3["S3 Bucket<br/>(Model Artifacts)"]
        EC2["EC2 Spot & On-Demand<br/>(Worker Nodes)"]
    end
    
    subgraph K8s["Kubernetes Cluster"]
        KubeRay["KubeRay Operator<br/>(Ray Cluster Manager)"]
        
        subgraph RayTraining["Ray Training Cluster"]
            TrainHead["Ray Head Node"]
            TrainWorkers["Ray Worker Nodes<br/>(4+ CPUs)"]
        end
        
        subgraph RayTuning["Ray Tuning Cluster"]
            TuneHead["Ray Head Node"]
            TuneWorkers["Ray Worker Nodes<br/>(4+ CPUs)"]
        end
        
        subgraph RayServing["Ray Serving Cluster"]
            ServeHead["Ray Head Node"]
            ServeWorkers["Ray Worker Nodes<br/>(0.5 CPU)"]
        end
        
        MLFlow["MLFlow + MinIO<br/>(Experiment Tracking)"]
        Prometheus["Prometheus + Grafana<br/>(Monitoring)"]
        ClusterAutoscaler["Cluster Autoscaler<br/>(Dynamic Scaling)"]
    end
    
    Client["Client<br/>(HTTP/gRPC)"]
    Developer["Developer<br/>(Ray Jobs)"]
    
    EKS --> K8s
    AWS --> EC2
    AWS --> S3
    
    Developer -->|Submit Job| KubeRay
    Client -->|Inference| RayServing
    
    RayTraining --> S3
    RayTuning --> S3
    RayServing -.->|Load Model| S3
    
    RayTraining --> MLFlow
    RayTuning --> MLFlow
    
    RayTraining --> Prometheus
    RayTuning --> Prometheus
    RayServing --> Prometheus
    
    EC2 -.->|Autoscale| ClusterAutoscaler

```

---

## Component Architecture

### 1. Infrastructure Layer (EKS & Kubernetes)

```mermaid
graph LR
    subgraph EKS["EKS Cluster (Kubernetes)"]
        Control["Control Plane<br/>(AWS Managed)"]
        
        subgraph NodeGroups["Node Groups"]
            OnDemand["On-Demand Nodes<br/>(Stable workloads)"]
            Spot["Spot Nodes<br/>(Cost-optimized)"]
        end
        
        subgraph SystemAddons["System Add-ons"]
            CNI["AWS CNI<br/>(Networking)"]
            CoreDNS["CoreDNS<br/>(Service Discovery)"]
            Metrics["Metrics Server<br/>(Resource Metrics)"]
        end
    end
    
    Control --> NodeGroups
    Control --> SystemAddons
    OnDemand -->|Labels: intention=training| Spot
```

**Key Infrastructure Decisions:**
- **EKS**: Managed Kubernetes on AWS
- **Node Types**:
  - On-Demand: For critical services (head nodes, control plane)
  - Spot: For worker nodes (80% cost savings)
- **Node Selector Strategy**: `intention=training` and `lifecycle=Ec2Spot` labels for workload affinity
- **Cluster Autoscaler**: Dynamically scales node groups based on pending pod requests

---

### 2. KubeRay Operator & Ray Clusters

```mermaid
graph TB
    subgraph Operator["KubeRay Operator<br/>(ray-system namespace)"]
        KubeRayController["KubeRay Controller<br/>(Reconciles RayCluster CRDs)"]
        MetricsServer["Metrics Server<br/>(Node resource tracking)"]
    end
    
    subgraph RayClusters["Ray Clusters (Custom Resources)"]
        TrainCRD["RayCluster: sentinel-training<br/>(ray-recommend-train-job.yaml)"]
        TuneCRD["RayCluster: sentinel-tuning<br/>(ray-recommend-tune-job.yaml)"]
        ServeCRD["RayService: sentinel-serving-recommend<br/>(ray-recommend-serve.yaml)"]
    end
    
    subgraph TrainInstance["Training Cluster Instance"]
        TH["Head Pod<br/>(1 CPU, 2Gi)"]
        TW1["Worker Pod 1<br/>(1 CPU, 2Gi)"]
        TW2["Worker Pod 2<br/>(1 CPU, 2Gi)"]
    end
    
    Operator -->|Watches & Reconciles| RayClusters
    TrainCRD -->|Creates| TrainInstance
    TH --> TW1
    TH --> TW2

```

**Ray Cluster Components:**
- **Head Node**: Scheduler, object store, dashboard (port 8265), HTTP gateway (port 8000)
- **Worker Nodes**: Execute tasks and store distributed objects
- **RayCluster CRD**: Kubernetes Custom Resource defining cluster topology
- **RayService CRD**: Wraps RayCluster + Ray Serve deployments (auto health checks)

---

### 3. Machine Learning Pipeline

```mermaid
graph LR
    subgraph DataOps["Data Pipeline"]
        MovieLens["MovieLens 100k<br/>(943 users, 1.6k items)"]
        Download["Download & Cache<br/>(Auto on first run)"]
    end
    
    subgraph Training["Ray Training<br/>(Distributed PyTorch)"]
        LoadData["Load MovieLensDataset<br/>(PyTorch DataLoader)"]
        MatrixFact["MatrixFactorization<br/>(PyTorch Lightning)"]
        Train["Train Loop<br/>(5 epochs)"]
    end
    
    subgraph Tuning["Ray Tune<br/>(Hyperparameter Optimization)"]
        ASHA["ASHA Scheduler<br/>(Early Stopping)"]
        Trials["Trial Runs<br/>(batch_size, lr, embedding_dim)"]
    end
    
    subgraph Artifact["Model Storage"]
        S3Store["S3 Checkpoints<br/>(Best + Last models)"]
    end
    
    MovieLens --> Download
    Download --> LoadData
    LoadData --> MatrixFact
    MatrixFact --> Train
    Train --> S3Store
    
    Train -->|Metrics| ASHA
    ASHA --> Trials
    Trials --> S3Store

```

**Pipeline Stages:**
1. **Data Ingestion**: MovieLens 100k automatically downloaded on first run
2. **Training**: Distributed PyTorch via Ray Train + Lightning
3. **Hyperparameter Tuning**: Ray Tune with ASHA early stopping (~80% compute savings)
4. **Artifact Storage**: Checkpoints saved to S3 for reproducibility

---

### 4. Serving Layer (Ray Serve)

```mermaid
graph TB
    subgraph Deployment["Ray Serve Deployment"]
        RayServe["RayServing Cluster<br/>(ray-recommend-serve.yaml)"]
    end
    
    subgraph Endpoint["HTTP Endpoint"]
        Gateway["Ray Serve Gateway<br/>(Port 8000)"]
        Router["Route Prefix: /recommend"]
    end
    
    subgraph ReplicaPool["Replica Pool (Auto-scaled)"]
        Replica1["MovieRecommender Replica 1<br/>(Load Model from S3)"]
        Replica2["MovieRecommender Replica 2"]
        Replica3["MovieRecommender Replica N"]
    end
    
    subgraph Config["Scaling Policy"]
        AutoScale["Autoscaling:<br/>min=1, max=5 replicas<br/>target_requests=5 per replica<br/>upscale_delay=1s<br/>downscale_delay=30s"]
    end
    
    Deployment --> Endpoint
    Endpoint --> Router
    Router --> ReplicaPool
    ReplicaPool --> Config
    
    Replica1 -.->|Load| S3["S3<br/>(Checkpoint)"]

```

**Serving Features:**
- **HTTP Gateway**: TensorFlow Serving-style REST API
- **Auto-scaling**: Policies scale replicas 1-5 based on pending request queue
- **Lazy Loading**: Models downloaded from S3 on pod startup
- **Stateless**: Each replica is independent; easy horizontal scaling

---

### 5. Monitoring & Observability

```mermaid
graph TB
    subgraph Clusters["Ray Clusters"]
        Train["Training Cluster"]
        Tune["Tuning Cluster"]
        Serve["Serving Cluster"]
    end
    
    subgraph Monitoring["Monitoring Stack"]
        Prometheus["Prometheus<br/>(Metrics Collection)"]
        Grafana["Grafana<br/>(Dashboards)"]
        Prom_Operator["Prometheus Operator<br/>(CRD Management)"]
    end
    
    subgraph RayMetrics["Ray Metrics"]
        RayMonitor["Ray Monitor<br/>(ray-monitor.yaml)<br/>ServiceMonitor CRD"]
    end
    
    subgraph Dashboards["Dashboards"]
        TrainDash["Training Dashboard"]
        ServeDash["Serving Dashboard"]
    end
    
    Train --> RayMetrics
    Tune --> RayMetrics
    Serve --> RayMetrics
    
    RayMetrics -->|Scrapes| Prometheus
    Prometheus --> Grafana
    Prom_Operator -->|Configures| Prometheus
    
    Grafana --> TrainDash
    Grafana --> ServeDash

```

**Monitoring Components:**
- **Prometheus**: Scrapes Ray metrics (CPU, memory, task queue length)
- **Grafana**: Visualizes training progress and serving load
- **ServiceMonitor**: Kubernetes-native Prometheus scrape configuration
- **Ray Metrics**: Built-in Ray metrics exporters for all clusters

---

## Deployment Manifests Reference

### 1. **bin/kuberay-operator.sh**
Installs KubeRay operator and metrics server:
```bash
helm install kuberay-operator kuberay/kuberay-operator --version 1.5.1
helm install metrics-server metrics-server/metrics-server --version 3.13.0
```

### 2. **bin/custom-image.sh**
Builds and pushes Docker images:
- `Dockerfile.serve`: Ray Serve runtime with model inference code
- `Dockerfile.train`: Ray Train runtime with PyTorch Lightning

### 3. **bin/prometheus.sh**
Installs Prometheus stack and configures Ray scraping:
```bash
helm install prometheus prometheus-community/kube-prometheus-stack
kubectl apply -f k8s/ray-monitor.yaml
```

---

## Kubernetes Manifests

```mermaid
graph LR
    subgraph Manifests["k8s/ Directory"]
        RA["ray-recommend-train-job.yaml<br/>(Training RayJob)"]
        RU["ray-recommend-tune-job.yaml<br/>(Tuning RayJob)"]
        RS["ray-recommend-serve.yaml<br/>(Serving RayService)"]
        
        CA["cluster-autoscaler.yaml<br/>(DaemonSet)"]
        ML["mlflow-platform.yaml<br/>(MLFlow + MinIO)"]
        MON["ray-monitor.yaml<br/>(Prometheus ServiceMonitor)"]
    end
    
    subgraph Applied["Applied to Cluster"]
        Jobs["RayJob Resources"]
        Services["RayService Resources"]
        System["System Add-ons"]
    end
    
    RA --> Jobs
    RU --> Jobs
    RS --> Services
    CA --> System
    ML --> System
    MON --> System

```

---

## Data Flow Diagram

### Training & Tuning Pipeline
```mermaid
sequenceDiagram
    Developer->>+KubeRay: Submit RayJob (train_recommend.py)
    KubeRay->>+EKS: Create Ray Cluster Pods
    EKS->>+RayCluster: Schedule Head + 4 Worker Pods
    RayCluster->>MovieLens: Download Dataset
    RayCluster->>+Training: Parallel SGD (4 workers)
    Training->>Training: Compute loss on shards
    Training->>S3: Save checkpoint every epoch
    Training->>MLFlow: Log metrics (loss, accuracy)
    Training-->>-KubeRay: Job Complete
    KubeRay-->>-Developer: Return checkpoint path

```

### Serving Request Flow
```mermaid
sequenceDiagram
    Client->>+RayServe: POST /recommend<br/>{"user_id": 42, "movie_ids": [1,2,3]}
    RayServe->>RayServe: Route to MovieRecommender
    RayServe->>Replica: Forward request
    Replica->>+S3: Load model.ckpt (if not cached)
    S3-->>-Replica: Model bytes
    Replica->>+MatrixFactorization: Forward pass
    MatrixFactorization-->>-Replica: Predictions [4.2, 3.8, 2.5]
    Replica-->>RayServe: Response
    RayServe-->>-Client: 200 OK<br/>{"user_id": 42,<br/>"predictions": [...]}

```

---

## Scaling Strategy

### Horizontal Scaling (Replicas)

| Scenario | Config | Behavior |
|----------|--------|----------|
| Low traffic (0-5 req/s) | min_replicas=1 | Single replica handles all requests |
| Medium traffic (5-25 req/s) | max_replicas=5 | Scale to 5 replicas (1 req per replica) |
| Traffic spike | upscale_delay_s=1 | Add replicas within 1 second |
| Traffic drop | downscale_delay_s=30 | Remove unused replicas after 30s idle |

### Vertical Scaling (Node Capacity)

| Component | CPU/Pod | Memory/Pod | Purpose |
|-----------|---------|-----------|---------|
| Head Node | 0.5-1.0 | 2Gi | Scheduling, dashboard, object store |
| Training Worker | 1.0 | 2Gi | Distributed SGD computation |
| Serving Worker | 0.5 | 2Gi | Model inference (lightweight) |

### Node Group Scaling (Cluster Autoscaler)

```
Pending Pod detected
  ↓
Cluster Autoscaler checks available nodes
  ↓
No capacity? → Scale up node group (EC2 Spot/On-Demand)
  ↓
Node ready → Kubelet schedules pod
```

---

## Security & IAM

```mermaid
graph TB
    EKS["EKS Cluster"]
    IRSA["IAM Roles for Service Accounts<br/>(IRSA)"]
    S3["S3 Bucket<br/>(training-artifacts-du-yuyang)"]
    
    EKS -->|AssumesRole| IRSA
    IRSA -->|S3:GetObject| S3
    IRSA -->|S3:PutObject| S3

```

**IAM Permissions:**
- Ray Serve pods assume IAM role → download models from S3
- Training pods assume IAM role → write checkpoints to S3
- Principle: Least privilege (S3 bucket-scoped, not wildcard)

---

## Networking

```mermaid
graph TB
    subgraph K8sNetwork["Kubernetes Network"]
        CNI["AWS VPC CNI<br/>(Pod networking)"]
        SVC["Kubernetes Services<br/>(Service Discovery)"]
    end
    
    subgraph RayComm["Ray Internal Communication"]
        Raylet["Raylet<br/>(Worker agent)"]
        ObjectStore["Ray Object Store<br/>(Distributed memory)"]
        Scheduler["Ray Scheduler<br/>(Task assignment)"]
    end
    
    CNI -.->|Provides IP space| RayComm
    SVC -.->|Exposes| Raylet
    SVC -.->|Exposes| Scheduler

```

**Network Design:**
- **AWS CNI**: Each pod gets IP from VPC subnets
- **Ray Raylet**: Inter-pod communication on custom ports
- **Object Store**: Redis-backed distributed shared memory
- **Service Mesh** (optional): Could add Istio for traffic splitting

---

## Development Workflow

```mermaid
graph LR
    Dev["Developer<br/>(Local Machine)"]
    
    subgraph Local["Local Development"]
        Code["Python Code<br/>(train_recommend.py)"]
        Test["Unit Tests"]
        Mock["Mock Ray Cluster"]
    end
    
    subgraph CI["CI/CD Pipeline<br/>(GitHub Actions)"]
        Build["Build Docker Image"]
        Scan["Security Scan"]
        Push["Push to ECR"]
    end
    
    subgraph Staging["Staging Cluster<br/>(kind or EKS)"]
        Deploy["kubectl apply<br/>-f k8s/"]
        Validate["Run Integration Tests"]
    end
    
    subgraph Prod["Production Cluster<br/>(EKS)"]
        Submit["kubectl apply<br/>-f k8s/"]
        Monitor["Prometheus Dashboards"]
    end
    
    Dev -->|Write & Test| Local
    Local -->|Git Push| CI
    CI -->|Image| Staging
    Staging -->|Promote| Prod
    Prod -->|Monitor| Monitor

```

---

## Troubleshooting Checklist

### Pod Scheduling Issues
```
❌ "0/4 nodes are available: Insufficient cpu"
✅ Solutions:
  - Reduce CPU requests (e.g., 1.0 → 0.5)
  - Scale node group (cluster autoscaler)
  - Check node selector constraints

❌ "Pod didn't match node affinity/selector"
✅ Solutions:
  - Verify node labels: kubectl get nodes --show-labels
  - Check nodeSelector/affinity in manifest
  - Add labels to nodes: kubectl label nodes...
```

### Model Loading Failures
```
❌ "S3 access denied"
✅ Solutions:
  - Verify IRSA permissions
  - Check S3 bucket policy
  - Ensure checkpoint path is correct

❌ "Model not found at s3://..."
✅ Solutions:
  - List S3 checkpoints: aws s3 ls s3://training-artifacts-du-yuyang/
  - Update RayService manifest with correct path
```

### Monitoring Issues
```
❌ "Prometheus not scraping Ray metrics"
✅ Solutions:
  - Check ServiceMonitor: kubectl get servicemonitor -n monitoring
  - Verify ray-monitor.yaml applied
  - Port-forward to check metrics: kubectl port-forward...
```

---

## References

- **KubeRay Docs**: https://docs.ray.io/en/latest/kuberay/index.html
- **Ray Train**: https://docs.ray.io/en/latest/train/train.html
- **Ray Serve**: https://docs.ray.io/en/latest/serve/index.html
- **Ray Tune**: https://docs.ray.io/en/latest/tune/index.html
- **EKS Best Practices**: https://aws.github.io/aws-eks-best-practices/

