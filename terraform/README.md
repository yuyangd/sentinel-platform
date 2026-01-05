# Sentinel Platform - Terraform Infrastructure

This directory contains Terraform configuration files to provision AWS resources for the Sentinel MLOps Platform, including:

- **EKS Cluster** - Managed Kubernetes cluster with managed and spot node groups
- **OIDC Provider** - OpenID Connect provider for IRSA (IAM Roles for Service Accounts)
- **IAM Roles & Policies** - Cluster, node group, and service account roles
- **S3 Bucket** - Training artifacts storage with encryption and versioning
- **Kubernetes Resources** - Namespace and service account with IRSA annotations

## Prerequisites

- AWS CLI configured with appropriate credentials
- Terraform >= 1.0
- kubectl
- AWS Account with sufficient permissions
- Existing VPC and subnets (values configured in `terraform.tfvars`)

## Quick Start

### 1. Initialize Terraform

```bash
terraform init
```

### 2. Create terraform.tfvars

Copy the example and update with your values:

```bash
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` with your specific AWS account details and VPC configuration:

```hcl
aws_region = "ap-southeast-2"
cluster_name = "du-yuyang-training"
vpc_id = "vpc-8b188eee"

public_subnets = {
  "ap-southeast-2a" = "subnet-a27806c7"
  "ap-southeast-2b" = "subnet-55d7b322"
}

private_subnets = {
  "ap-southeast-2a" = "subnet-87e897e2"
  "ap-southeast-2b" = "subnet-54d7b323"
}

s3_bucket_name = "training-artifacts-du-yuyang"
```

### 3. Plan the deployment

```bash
terraform plan -out=tfplan
```

Review the plan to ensure all resources will be created as expected.

### 4. Apply the configuration

```bash
terraform apply tfplan
```

This will create:
- EKS cluster with 2 node groups (managed + spot)
- OIDC provider for IRSA
- S3 bucket with encryption, versioning, and access logging
- IAM roles and policies
- Kubernetes namespace and service account

### 5. Configure kubectl

After Terraform completes, configure kubectl to access the cluster:

```bash
aws eks update-kubeconfig --name du-yuyang-training --region ap-southeast-2
```

Or use the output from Terraform:

```bash
terraform output -raw configure_kubectl
```

## File Structure

- `provider.tf` - AWS and Kubernetes provider configuration
- `variables.tf` - Input variables with defaults
- `eks.tf` - EKS cluster, node groups, and OIDC provider
- `iam.tf` - IAM roles, policies, and Kubernetes resources
- `s3.tf` - S3 bucket configuration with security best practices
- `outputs.tf` - Output values for cluster access
- `terraform.tfvars.example` - Example terraform variables

## Key Features

### Security Best Practices

- ✅ Encryption at rest (S3, EBS)
- ✅ Access logging (S3)
- ✅ Versioning enabled (S3)
- ✅ Public access blocked (S3)
- ✅ Private networking for nodes
- ✅ IRSA (IAM Roles for Service Accounts) - no hardcoded credentials
- ✅ OIDC-based authentication for service accounts

### Cost Optimization

- ✅ Spot instances for training workloads
- ✅ Multi-instance type flexibility for spot nodes
- ✅ Configurable scaling parameters
- ✅ Private subnets reduce NAT gateway costs

## Useful Commands

### View current state

```bash
terraform state list
terraform state show aws_eks_cluster.main
```

### Get outputs

```bash
terraform output
terraform output eks_cluster_endpoint
terraform output ray_service_account_role_arn
```

### Scale nodes

To change the desired capacity of node groups, update the `terraform.tfvars` and apply:

```bash
# Edit terraform.tfvars
# managed_node_group_desired_capacity = 3
# spot_node_group_desired_capacity = 2

terraform plan
terraform apply
```

### Destroy resources

⚠️ **Warning**: This will delete the cluster and all data (except S3 with versioning).

```bash
terraform destroy
```

## Troubleshooting

### kubectl connection fails

```bash
# Update kubeconfig
aws eks update-kubeconfig --name du-yuyang-training --region ap-southeast-2

# Test connection
kubectl get nodes
```

### S3 bucket already exists

Terraform will error if the bucket name already exists. S3 bucket names are globally unique.

Change `s3_bucket_name` in `terraform.tfvars` to a unique name.

### Node group fails to provision

Check CloudWatch logs:

```bash
aws logs describe-log-groups --region ap-southeast-2
```

### OIDC provider issues

The OIDC provider URL is automatically derived from the EKS cluster. If you encounter issues with IRSA:

```bash
# Verify OIDC provider
aws iam list-open-id-connect-providers

# Check service account annotation
kubectl get sa ray-service-account -n sentinel-prod -o jsonpath='{.metadata.annotations}'
```

## Integration with Existing Resources

This Terraform configuration assumes:

1. VPC and subnets already exist
2. Subnets are tagged for ELB (if using load balancers):
   ```bash
   aws ec2 create-tags --resources subnet-xxxxx --tags Key=kubernetes.io/role/elb,Value=1
   ```

3. IAM user has permissions to create EKS clusters, IAM roles, and S3 buckets

## Next Steps

After provisioning:

1. Install KubeRay operator:
   ```bash
   helm repo add kuberay https://ray-project.github.io/kuberay-helm/
   helm install kuberay-operator kuberay/kuberay-operator --namespace kuberay-system --create-namespace
   ```

2. Deploy MLflow:
   ```bash
   kubectl apply -f ../k8s/mlflow/mlflow-platform.yaml -n sentinel-prod
   ```

3. Deploy Ray clusters:
   ```bash
   kubectl apply -f ../k8s/ray/ray-cluster-train.yaml -n sentinel-prod
   ```

## Support

For issues or questions, refer to:

- [Terraform AWS Provider Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [EKS Terraform Module](https://github.com/terraform-aws-modules/terraform-aws-eks)
- [AWS EKS Documentation](https://docs.aws.amazon.com/eks/)
