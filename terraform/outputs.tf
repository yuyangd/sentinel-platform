################################################################################
# EKS Outputs
################################################################################

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = aws_eks_cluster.main.name
}

output "eks_cluster_endpoint" {
  description = "EKS cluster API endpoint"
  value       = aws_eks_cluster.main.endpoint
}

output "eks_cluster_version" {
  description = "EKS cluster version"
  value       = aws_eks_cluster.main.version
}

output "eks_cluster_platform_version" {
  description = "EKS cluster platform version"
  value       = aws_eks_cluster.main.platform_version
}

output "eks_cluster_certificate_authority" {
  description = "EKS cluster certificate authority"
  value       = aws_eks_cluster.main.certificate_authority[0].data
  sensitive   = true
}

output "eks_cluster_arn" {
  description = "EKS cluster ARN"
  value       = aws_eks_cluster.main.arn
}

################################################################################
# OIDC Provider Outputs
################################################################################

output "oidc_provider_arn" {
  description = "ARN of the OIDC Provider"
  value       = aws_iam_openid_connect_provider.eks.arn
}

output "oidc_provider_url" {
  description = "URL of the OIDC Provider"
  value       = aws_iam_openid_connect_provider.eks.url
}

################################################################################
# Node Group Outputs
################################################################################

output "managed_node_group_id" {
  description = "Managed node group ID"
  value       = aws_eks_node_group.managed.id
}

output "managed_node_group_arn" {
  description = "Managed node group ARN"
  value       = aws_eks_node_group.managed.arn
}

output "spot_node_group_id" {
  description = "Spot node group ID"
  value       = aws_eks_node_group.spot.id
}

output "spot_node_group_arn" {
  description = "Spot node group ARN"
  value       = aws_eks_node_group.spot.arn
}

################################################################################
# IAM Outputs
################################################################################

output "eks_cluster_role_arn" {
  description = "ARN of EKS cluster IAM role"
  value       = aws_iam_role.eks_cluster_role.arn
}

output "node_group_role_arn" {
  description = "ARN of node group IAM role"
  value       = aws_iam_role.node_group_role.arn
}

output "ray_service_account_role_arn" {
  description = "ARN of Ray service account IAM role"
  value       = aws_iam_role.ray_service_account.arn
}

output "s3_access_policy_arn" {
  description = "ARN of S3 access policy"
  value       = aws_iam_policy.s3_access.arn
}

################################################################################
# S3 Outputs
################################################################################

output "s3_bucket_name" {
  description = "Name of the S3 training artifacts bucket"
  value       = aws_s3_bucket.training_artifacts.id
}

output "s3_bucket_arn" {
  description = "ARN of the S3 training artifacts bucket"
  value       = aws_s3_bucket.training_artifacts.arn
}

output "s3_bucket_logs_name" {
  description = "Name of the S3 logs bucket"
  value       = aws_s3_bucket.training_artifacts_logs.id
}

################################################################################
# Kubernetes Outputs
################################################################################

output "kubernetes_namespace" {
  description = "Kubernetes namespace for Sentinel"
  value       = kubernetes_namespace.sentinel.metadata[0].name
}

output "ray_service_account_name" {
  description = "Kubernetes service account name for Ray"
  value       = kubernetes_service_account.ray.metadata[0].name
}

################################################################################
# Configure kubectl
################################################################################

output "configure_kubectl" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --name ${aws_eks_cluster.main.name} --region ${var.aws_region}"
}
