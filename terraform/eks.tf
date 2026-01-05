################################################################################
# EKS Cluster
################################################################################

resource "aws_eks_cluster" "main" {
  name     = var.cluster_name
  role_arn = aws_iam_role.eks_cluster_role.arn
  version  = "1.27" # Specify your desired K8s version

  vpc_config {
    subnet_ids              = concat(values(var.public_subnets), values(var.private_subnets))
    endpoint_private_access = true
    endpoint_public_access  = true
  }

  enabled_cluster_log_types = var.enable_cloudwatch_logging ? [
    "api",
    "audit",
    "authenticator",
    "controllerManager",
    "scheduler"
  ] : []

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
    aws_iam_role_policy_attachment.eks_vpc_resource_controller,
    aws_cloudwatch_log_group.eks_cluster
  ]

  tags = {
    Name = var.cluster_name
  }
}

# CloudWatch Log Group for EKS
resource "aws_cloudwatch_log_group" "eks_cluster" {
  count             = var.enable_cloudwatch_logging ? 1 : 0
  name              = "/aws/eks/${var.cluster_name}/cluster"
  retention_in_days = var.eks_log_retention_days

  tags = {
    Name = "eks-${var.cluster_name}"
  }
}

# Data source to get auth token
data "aws_eks_cluster_auth" "cluster" {
  name = aws_eks_cluster.main.name
}

################################################################################
# EKS Cluster IAM Role
################################################################################

resource "aws_iam_role" "eks_cluster_role" {
  name               = "${var.cluster_name}-cluster-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "${var.cluster_name}-cluster-role"
  }
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_cluster_role.name
}

resource "aws_iam_role_policy_attachment" "eks_vpc_resource_controller" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSVPCResourceController"
  role       = aws_iam_role.eks_cluster_role.name
}

################################################################################
# OIDC Provider
################################################################################

resource "aws_iam_openid_connect_provider" "eks" {
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.eks.certificates[0].sha1_fingerprint]
  url             = aws_eks_cluster.main.identity[0].oidc[0].issuer
}

data "tls_certificate" "eks" {
  url = aws_eks_cluster.main.identity[0].oidc[0].issuer
}

################################################################################
# Managed Node Group
################################################################################

resource "aws_eks_node_group" "managed" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "managed-ng-1"
  node_role_arn   = aws_iam_role.node_group_role.arn
  subnet_ids      = values(var.private_subnets)
  version         = aws_eks_cluster.main.version

  scaling_config {
    desired_size = var.managed_node_group_desired_capacity
    max_size     = var.managed_node_group_max_size
    min_size     = var.managed_node_group_min_size
  }

  instance_types = [var.managed_node_group_instance_type]
  disk_size      = var.managed_node_group_volume_size

  tags = {
    Name = "managed-ng-1"
  }

  depends_on = [
    aws_iam_role_policy_attachment.node_group_policy,
    aws_iam_role_policy_attachment.node_group_cni_policy,
    aws_iam_role_policy_attachment.node_group_container_registry_policy,
    aws_iam_role_policy_attachment.node_group_ssm_policy,
  ]

  lifecycle {
    create_before_destroy = true
  }
}

################################################################################
# Spot Node Group
################################################################################

resource "aws_eks_node_group" "spot" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "worker-group-spot-1"
  node_role_arn   = aws_iam_role.node_group_role.arn
  subnet_ids      = values(var.private_subnets)
  version         = aws_eks_cluster.main.version

  scaling_config {
    desired_size = var.spot_node_group_desired_capacity
    max_size     = var.spot_node_group_max_size
    min_size     = var.spot_node_group_min_size
  }

  instance_types = var.spot_node_group_instance_types
  capacity_type  = "SPOT"
  disk_size      = var.managed_node_group_volume_size

  labels = {
    lifecycle  = "Ec2Spot"
    intention  = "training"
  }

  tags = {
    Name = "worker-group-spot-1"
  }

  depends_on = [
    aws_iam_role_policy_attachment.node_group_policy,
    aws_iam_role_policy_attachment.node_group_cni_policy,
    aws_iam_role_policy_attachment.node_group_container_registry_policy,
    aws_iam_role_policy_attachment.node_group_ssm_policy,
  ]

  lifecycle {
    create_before_destroy = true
  }
}

################################################################################
# Node Group IAM Role
################################################################################

resource "aws_iam_role" "node_group_role" {
  name               = "${var.cluster_name}-node-group-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "${var.cluster_name}-node-group-role"
  }
}

resource "aws_iam_role_policy_attachment" "node_group_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.node_group_role.name
}

resource "aws_iam_role_policy_attachment" "node_group_cni_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.node_group_role.name
}

resource "aws_iam_role_policy_attachment" "node_group_container_registry_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.node_group_role.name
}

resource "aws_iam_role_policy_attachment" "node_group_ssm_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
  role       = aws_iam_role.node_group_role.name
}
