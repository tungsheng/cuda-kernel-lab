variable "aws_region" {
  description = "AWS region for the disposable GPU benchmark host."
  type        = string
  default     = "us-west-2"
}

variable "name" {
  description = "Name tag for the benchmark instance, VPC, and EC2-module-managed security group."
  type        = string
  default     = "cuda-kernel-lab-gpu"
}

variable "instance_type" {
  description = "EC2 GPU instance type."
  type        = string
  default     = "g5.xlarge"
}

variable "key_name" {
  description = "Optional EC2 key pair name used for SSH access."
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
  description = "Existing VPC ID. When omitted with subnet_id, Terraform creates a disposable VPC."
  type        = string
  default     = null
}

variable "subnet_id" {
  description = "Existing subnet ID. Defaults to the disposable VPC public subnet, or the first subnet in vpc_id when provided."
  type        = string
  default     = null
}

variable "vpc_cidr" {
  description = "CIDR block for the disposable VPC created when vpc_id and subnet_id are omitted."
  type        = string
  default     = "10.42.0.0/16"
}

variable "public_subnet_cidr" {
  description = "CIDR block for the disposable public subnet created by the VPC module."
  type        = string
  default     = "10.42.1.0/24"
}

variable "security_group_id" {
  description = "Existing security group ID. When omitted, the EC2 module creates a temporary SSH security group."
  type        = string
  default     = null
}

variable "ssh_ingress_cidr" {
  description = "CIDR allowed to reach SSH when the EC2 module creates the security group."
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
