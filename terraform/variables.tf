variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-southeast-2"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "du-yuyang-training"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "training"
}

variable "owner" {
  description = "Owner of the resources"
  type        = string
  default     = "du.y"
}

variable "rea_system_id" {
  description = "REA system ID"
  type        = string
  default     = "x-du-yuyang"
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
  default     = "vpc-8b188eee"
}

variable "public_subnets" {
  description = "Public subnet IDs"
  type        = map(string)
  default = {
    "ap-southeast-2a" = "subnet-a27806c7"
    "ap-southeast-2b" = "subnet-55d7b322"
  }
}

variable "private_subnets" {
  description = "Private subnet IDs"
  type        = map(string)
  default = {
    "ap-southeast-2a" = "subnet-87e897e2"
    "ap-southeast-2b" = "subnet-54d7b323"
  }
}

variable "managed_node_group_instance_type" {
  description = "Instance type for managed node group"
  type        = string
  default     = "t4g.large"
}

variable "managed_node_group_min_size" {
  description = "Minimum size of managed node group"
  type        = number
  default     = 2
}

variable "managed_node_group_max_size" {
  description = "Maximum size of managed node group"
  type        = number
  default     = 3
}

variable "managed_node_group_desired_capacity" {
  description = "Desired capacity of managed node group"
  type        = number
  default     = 2
}

variable "managed_node_group_volume_size" {
  description = "Volume size for managed node group"
  type        = number
  default     = 50
}

variable "spot_node_group_instance_types" {
  description = "Instance types for spot node group"
  type        = list(string)
  default     = ["t4g.medium", "t4g.large"]
}

variable "spot_node_group_min_size" {
  description = "Minimum size of spot node group"
  type        = number
  default     = 0
}

variable "spot_node_group_max_size" {
  description = "Maximum size of spot node group"
  type        = number
  default     = 5
}

variable "spot_node_group_desired_capacity" {
  description = "Desired capacity of spot node group"
  type        = number
  default     = 0
}

variable "s3_bucket_name" {
  description = "S3 bucket name for training artifacts"
  type        = string
  default     = "training-artifacts-du-yuyang"
}

variable "kubernetes_namespace" {
  description = "Kubernetes namespace for Ray service account"
  type        = string
  default     = "sentinel-prod"
}

variable "ray_service_account_name" {
  description = "Kubernetes service account name for Ray"
  type        = string
  default     = "ray-service-account"
}

variable "enable_cluster_autoscaling" {
  description = "Enable cluster autoscaling"
  type        = bool
  default     = true
}

variable "enable_cloudwatch_logging" {
  description = "Enable CloudWatch logging for EKS control plane"
  type        = bool
  default     = true
}

variable "eks_log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 7
}
