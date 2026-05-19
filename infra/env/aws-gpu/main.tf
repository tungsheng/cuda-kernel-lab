terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_ssm_parameter" "gpu_ami" {
  name = var.ami_ssm_parameter
}

data "aws_vpc" "default" {
  count = var.vpc_id == null && var.subnet_id == null ? 1 : 0

  default = true
}

data "aws_subnet" "selected" {
  count = var.subnet_id == null ? 0 : 1

  id = var.subnet_id
}

data "aws_subnets" "default" {
  count = var.subnet_id == null ? 1 : 0

  filter {
    name   = "vpc-id"
    values = [local.vpc_id]
  }

  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

locals {
  vpc_id = var.vpc_id != null ? var.vpc_id : (
    var.subnet_id != null ? data.aws_subnet.selected[0].vpc_id : data.aws_vpc.default[0].id
  )

  subnet_id = var.subnet_id != null ? var.subnet_id : sort(data.aws_subnets.default[0].ids)[0]

  common_tags = merge(
    var.tags,
    {
      Project = "cuda-kernel-lab"
      Purpose = "gpu-benchmark"
    }
  )

  security_group_id = var.security_group_id != null ? var.security_group_id : aws_security_group.ssh[0].id
}

resource "aws_security_group" "ssh" {
  count = var.security_group_id == null ? 1 : 0

  name_prefix = "${var.name}-ssh-"
  description = "SSH access for disposable cuda-kernel-lab GPU benchmark host"
  vpc_id      = local.vpc_id

  ingress {
    description = "benchmark SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [coalesce(var.ssh_ingress_cidr, "127.0.0.1/32")]
  }

  egress {
    description      = "HTTPS outbound"
    from_port        = 443
    to_port          = 443
    protocol         = "tcp"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }

  egress {
    description      = "HTTP outbound"
    from_port        = 80
    to_port          = 80
    protocol         = "tcp"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }

  egress {
    description      = "DNS UDP outbound"
    from_port        = 53
    to_port          = 53
    protocol         = "udp"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }

  egress {
    description      = "DNS TCP outbound"
    from_port        = 53
    to_port          = 53
    protocol         = "tcp"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${var.name}-ssh"
  })

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_instance" "gpu" {
  ami                         = data.aws_ssm_parameter.gpu_ami.value
  instance_type               = var.instance_type
  key_name                    = var.key_name
  subnet_id                   = local.subnet_id
  vpc_security_group_ids      = [local.security_group_id]
  associate_public_ip_address = var.associate_public_ip

  root_block_device {
    volume_size           = var.volume_size_gb
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true

    tags = merge(local.common_tags, {
      Name = var.name
    })
  }

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  tags = merge(local.common_tags, {
    Name = var.name
  })
}
