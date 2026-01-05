################################################################################
# S3 Bucket for Training Artifacts
################################################################################

resource "aws_s3_bucket" "training_artifacts" {
  bucket = var.s3_bucket_name

  tags = {
    Name = var.s3_bucket_name
  }
}

# Block public access
resource "aws_s3_bucket_public_access_block" "training_artifacts" {
  bucket = aws_s3_bucket.training_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Enable versioning for data protection
resource "aws_s3_bucket_versioning" "training_artifacts" {
  bucket = aws_s3_bucket.training_artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Enable server-side encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "training_artifacts" {
  bucket = aws_s3_bucket.training_artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
