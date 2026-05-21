# AWS GPU Terraform Environment

This environment creates the disposable single-node EC2 GPU host used for CUDA
Kernel Lab benchmark evidence.

Use the host lifecycle scripts for benchmark evidence:

```bash
./scripts/up
./scripts/benchmark --run-id <run-id>
./scripts/down
```

The default `./scripts/up` creates or reuses a project-local EC2 key pair and
private key under `.aws-gpu/keys/`, then writes `.aws-gpu/connection.env` for
`./scripts/benchmark`.

Use explicit key arguments only when you want to align with an existing EC2 key
pair:

```bash
./scripts/up --key-name <key-pair-name> --key-file <key-file.pem>
./scripts/down
```

Like the GPU nodes in `gpu-inference-lab`, this environment can launch without
an EC2 SSH key. Use `./scripts/up --skip-bootstrap` for an infra-only host
without SSH bootstrap.

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
- networking: disposable VPC module with one public subnet unless `vpc_id` or
  `subnet_id` is set
- modules: `terraform-aws-modules/vpc/aws` and
  `terraform-aws-modules/ec2-instance/aws`

The EC2 module creates an SSH security group when `security_group_id` is
omitted. Set `ssh_ingress_cidr` to your public IP CIDR for that path.
