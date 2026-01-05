################################################################################
# S3 Access Policy
################################################################################

resource "aws_iam_policy" "s3_access" {
  name        = "SentinelS3Access"
  description = "Policy for S3 access to training artifacts bucket"
  policy      = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.s3_bucket_name}"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.s3_bucket_name}/*"
        ]
      }
    ]
  })

  tags = {
    Name = "SentinelS3Access"
  }
}

################################################################################
# Ray Service Account IAM Role
################################################################################

locals {
  oidc_provider_arn = aws_iam_openid_connect_provider.eks.arn
  oidc_provider_url = replace(aws_iam_openid_connect_provider.eks.url, "https://", "")
}

resource "aws_iam_role" "ray_service_account" {
  name               = "${var.cluster_name}-ray-service-account"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = local.oidc_provider_arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${local.oidc_provider_url}:sub" = "system:serviceaccount:${var.kubernetes_namespace}:${var.ray_service_account_name}"
            "${local.oidc_provider_url}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = {
    Name = "${var.cluster_name}-ray-service-account"
  }
}

resource "aws_iam_role_policy_attachment" "ray_s3_access" {
  policy_arn = aws_iam_policy.s3_access.arn
  role       = aws_iam_role.ray_service_account.name
}

################################################################################
# Kubernetes Namespace
################################################################################

resource "kubernetes_namespace" "sentinel" {
  metadata {
    name = var.kubernetes_namespace
  }

  depends_on = [
    aws_eks_cluster.main
  ]
}

################################################################################
# Kubernetes Service Account
################################################################################

resource "kubernetes_service_account" "ray" {
  metadata {
    name      = var.ray_service_account_name
    namespace = kubernetes_namespace.sentinel.metadata[0].name
    annotations = {
      "eks.amazonaws.com/role-arn" = aws_iam_role.ray_service_account.arn
    }
  }

  depends_on = [
    kubernetes_namespace.sentinel
  ]
}
