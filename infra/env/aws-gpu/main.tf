terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 6.37, < 7.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_ssm_parameter" "gpu_ami" {
  name = var.ami_ssm_parameter
}

data "aws_availability_zones" "available" {
  count = local.create_vpc ? 1 : 0

  state = "available"
}

data "aws_subnet" "selected" {
  count = var.subnet_id == null ? 0 : 1

  id = var.subnet_id
}

data "aws_subnets" "selected" {
  count = !local.create_vpc && var.subnet_id == null ? 1 : 0

  filter {
    name   = "vpc-id"
    values = [var.vpc_id]
  }
}

locals {
  create_vpc = var.vpc_id == null && var.subnet_id == null

  common_tags = merge(
    var.tags,
    {
      Project = "cuda-kernel-lab"
      Purpose = "gpu-benchmark"
    }
  )

  vpc_id = (
    local.create_vpc
    ? module.vpc.vpc_id
    : (
      var.subnet_id != null
      ? data.aws_subnet.selected[0].vpc_id
      : var.vpc_id
    )
  )

  subnet_id = (
    local.create_vpc
    ? module.vpc.public_subnets[0]
    : (
      var.subnet_id != null
      ? var.subnet_id
      : sort(data.aws_subnets.selected[0].ids)[0]
    )
  )

  security_group_id  = var.security_group_id != null ? var.security_group_id : module.gpu_instance.security_group_id
  security_group_ids = var.security_group_id != null ? [var.security_group_id] : []
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "6.6.1"

  create_vpc = local.create_vpc
  name       = var.name
  cidr       = var.vpc_cidr

  azs                     = local.create_vpc ? [data.aws_availability_zones.available[0].names[0]] : []
  public_subnets          = local.create_vpc ? [var.public_subnet_cidr] : []
  enable_nat_gateway      = false
  map_public_ip_on_launch = var.associate_public_ip

  tags = local.common_tags
}

module "gpu_instance" {
  source  = "terraform-aws-modules/ec2-instance/aws"
  version = "6.4.0"

  name = var.name

  ami                         = data.aws_ssm_parameter.gpu_ami.value
  instance_type               = var.instance_type
  key_name                    = var.key_name
  subnet_id                   = local.subnet_id
  vpc_security_group_ids      = local.security_group_ids
  associate_public_ip_address = var.associate_public_ip
  create_security_group       = var.security_group_id == null
  security_group_name         = "${var.name}-ssh"
  security_group_description  = "SSH access for disposable cuda-kernel-lab GPU benchmark host"
  security_group_vpc_id       = local.vpc_id
  security_group_tags = merge(local.common_tags, {
    Name = "${var.name}-ssh"
  })

  security_group_ingress_rules = {
    ssh = {
      description = "benchmark SSH"
      cidr_ipv4   = coalesce(var.ssh_ingress_cidr, "127.0.0.1/32")
      from_port   = 22
      ip_protocol = "tcp"
      to_port     = 22
    }
  }

  security_group_egress_rules = {
    http_ipv4 = {
      cidr_ipv4   = "0.0.0.0/0"
      description = "HTTP outbound"
      from_port   = 80
      ip_protocol = "tcp"
      to_port     = 80
    }
    https_ipv4 = {
      cidr_ipv4   = "0.0.0.0/0"
      description = "HTTPS outbound"
      from_port   = 443
      ip_protocol = "tcp"
      to_port     = 443
    }
    dns_tcp_ipv4 = {
      cidr_ipv4   = "0.0.0.0/0"
      description = "DNS TCP outbound"
      from_port   = 53
      ip_protocol = "tcp"
      to_port     = 53
    }
    dns_udp_ipv4 = {
      cidr_ipv4   = "0.0.0.0/0"
      description = "DNS UDP outbound"
      from_port   = 53
      ip_protocol = "udp"
      to_port     = 53
    }
  }

  root_block_device = {
    delete_on_termination = true
    encrypted             = true
    size                  = var.volume_size_gb
    type                  = "gp3"
  }

  metadata_options = {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  tags = merge(local.common_tags, {
    Name = var.name
  })

  volume_tags = merge(local.common_tags, {
    Name = var.name
  })
}
