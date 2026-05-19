# AWS GPU Terraform Environment

This environment creates the disposable single-node EC2 GPU host used for CUDA
Kernel Lab benchmark evidence.

Use the wrapper scripts for the common path:

```bash
./scripts/up --key-name <key-pair-name> --key-file <key-file.pem>
./scripts/down
```

Use Terraform directly when you want to inspect or customize the plan:

```bash
terraform -chdir=infra/env/aws-gpu init
terraform -chdir=infra/env/aws-gpu plan \
  -out tfplan \
  -var 'key_name=<key-pair-name>' \
  -var 'ssh_ingress_cidr=<your-ip>/32'
terraform -chdir=infra/env/aws-gpu apply tfplan
```

Defaults:

- region: `us-west-2`
- instance type: `g5.xlarge`
- AMI source: AWS Deep Learning Base OSS NVIDIA Driver GPU Ubuntu 22.04 SSM
  parameter
- root volume: 100 GiB gp3, deleted on termination
- networking: default VPC/default subnet unless `vpc_id` or `subnet_id` is set

Terraform creates an SSH security group when `security_group_id` is omitted.
Set `ssh_ingress_cidr` to your public IP CIDR for that path.
