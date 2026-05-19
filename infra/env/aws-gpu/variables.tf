variable "aws_region" {
  description = "AWS region for the disposable GPU benchmark host."
  type        = string
  default     = "us-west-2"
}

variable "name" {
  description = "Name tag for the benchmark instance and Terraform-managed security group."
  type        = string
  default     = "cuda-kernel-lab-gpu"
}

variable "instance_type" {
  description = "EC2 GPU instance type."
  type        = string
  default     = "g5.xlarge"
}

variable "key_name" {
  description = "EC2 key pair name used for SSH access."
  type        = string
  default     = null
}

variable "ami_ssm_parameter" {
  description = "Public SSM parameter resolving to the latest AWS Deep Learning GPU AMI."
  type        = string
  default     = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
}

variable "volume_size_gb" {
  description = "Root EBS volume size in GiB."
  type        = number
  default     = 100

  validation {
    condition     = var.volume_size_gb >= 40
    error_message = "volume_size_gb must be at least 40."
  }
}

variable "associate_public_ip" {
  description = "Associate a public IPv4 address with the benchmark host."
  type        = bool
  default     = true
}

variable "vpc_id" {
  description = "Existing VPC ID. Defaults to the selected subnet VPC or the default VPC."
  type        = string
  default     = null
}

variable "subnet_id" {
  description = "Existing subnet ID. Defaults to the first default subnet in the selected/default VPC."
  type        = string
  default     = null
}

variable "security_group_id" {
  description = "Existing security group ID. When omitted, Terraform creates a temporary SSH security group."
  type        = string
  default     = null
}

variable "ssh_ingress_cidr" {
  description = "CIDR allowed to reach SSH when Terraform creates the security group."
  type        = string
  default     = null

  validation {
    condition     = var.ssh_ingress_cidr == null || !contains(["0.0.0.0/0", "::/0"], var.ssh_ingress_cidr)
    error_message = "ssh_ingress_cidr must not expose SSH to the entire internet."
  }
}

variable "tags" {
  description = "Extra tags to apply to benchmark resources."
  type        = map(string)
  default     = {}
}
