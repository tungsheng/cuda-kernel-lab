output "instance_id" {
  value = aws_instance.gpu.id
}

output "instance_type" {
  value = aws_instance.gpu.instance_type
}

output "ami_id" {
  value = nonsensitive(aws_instance.gpu.ami)
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
  value = try(aws_security_group.ssh[0].id, "")
}

output "public_ip" {
  value = aws_instance.gpu.public_ip
}

output "private_ip" {
  value = aws_instance.gpu.private_ip
}

output "ssh_host" {
  value = aws_instance.gpu.public_ip != "" ? aws_instance.gpu.public_ip : aws_instance.gpu.private_ip
}
