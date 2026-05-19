output "instance_id" {
  value = module.gpu_instance.id
}

output "instance_type" {
  value = var.instance_type
}

output "ami_id" {
  value = nonsensitive(module.gpu_instance.ami)
}

output "aws_region" {
  value = var.aws_region
}

output "vpc_id" {
  value = local.vpc_id
}

output "subnet_id" {
  value = local.subnet_id
}

output "security_group_id" {
  value = local.security_group_id
}

output "terraform_managed_security_group_id" {
  value = var.security_group_id == null ? module.gpu_instance.security_group_id : ""
}

output "public_ip" {
  value = module.gpu_instance.public_ip
}

output "private_ip" {
  value = module.gpu_instance.private_ip
}

output "ssh_host" {
  value = module.gpu_instance.public_ip != "" ? module.gpu_instance.public_ip : module.gpu_instance.private_ip
}
