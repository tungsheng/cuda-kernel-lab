# AWS GPU Terraform Environment

This environment creates the disposable single-node EC2 GPU host used by the
legacy AWS live-GPU workflow. Runpod is the default provider; always pass
`--platform aws` when using these lifecycle scripts.

## Script Flow

```bash
./scripts/up --platform aws
./scripts/benchmark --platform aws --run-id <run-id>
./scripts/down --platform aws
```

With no key arguments, `./scripts/up --platform aws` creates or reuses a
project-local EC2 key pair and private key under `.aws-gpu/keys/`, then writes
`.aws-gpu/connection.env` for `./scripts/benchmark --platform aws`.

Use explicit key arguments only when you need to align with an existing EC2 key
pair:

```bash
./scripts/up --platform aws --key-name <key-pair-name> --key-file <key-file.pem>
./scripts/down --platform aws
```

Use `./scripts/up --platform aws --skip-bootstrap` for an infra-only host
without SSH sync or dependency installation.

## Terraform Directly

Use Terraform directly only when you need to inspect or customize the plan:

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
- networking: disposable VPC module with one public subnet unless `vpc_id` or
  `subnet_id` is set
- modules: `terraform-aws-modules/vpc/aws` and
  `terraform-aws-modules/ec2-instance/aws`

The EC2 module creates an SSH security group when `security_group_id` is
omitted. Set `ssh_ingress_cidr` to your public IP CIDR for that path.
