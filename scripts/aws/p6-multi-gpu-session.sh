#!/usr/bin/env bash
# P6 multi-GPU AWS session (P6-01b runbook): one g6.12xlarge (4× NVIDIA L4,
# sm_89 — same arch as the RTX 4000 Ada dev box) in us-east-1 to prove the
# NCCL exchange correct on 2 and 4 real GPUs and to measure strong/weak
# scaling against the predictions in docs/perf/p6-multi-gpu.md.
#
# COST: ~$4.60/h on-demand; the instance self-terminates after 240 min, so the
# worst case is ≲ $19 plus a few cents of gp3. LAUNCH ONLY WITH THE USER'S
# EXPLICIT OK — this script is checked in, not run, by the PR that adds it.
#
# Usage (from the repo root, AWS CLI configured for us-east-1):
#   scripts/aws/p6-multi-gpu-session.sh launch     # prints IID / IP
#   scripts/aws/p6-multi-gpu-session.sh setup      # rustup + NCCL 2.32 + rsync
#   scripts/aws/p6-multi-gpu-session.sh run        # oracles + bench → results/p6-aws/
#   scripts/aws/p6-multi-gpu-session.sh teardown   # terminate + all-region scan
# State (IID, IP) is kept in ./.p6-aws.env; add it to .git/info/exclude so it
# is never committed. results/p6-aws/ collects topo, oracle and bench logs.
set -euo pipefail

REGION=us-east-1
TYPE=g6.12xlarge
KEY=aleph-p6
SG_NAME=aleph-p6
STATE=.p6-aws.env
SSH_OPTS=(-i "$HOME/.ssh/$KEY.pem" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
export AWS_DEFAULT_REGION=$REGION

say() { printf '\n== %s\n' "$*"; }
load() { [ -f "$STATE" ] && . "$STATE"; : "${IID:?no IID — run launch first}"; : "${IP:?}"; }
rsh() { ssh "${SSH_OPTS[@]}" "ubuntu@$IP" "$@"; }

launch() {
  say "AMI: DLAMI base (Ubuntu 22.04, NVIDIA driver) from SSM"
  AMI=$(aws ssm get-parameter \
    --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
    --query Parameter.Value --output text)
  echo "AMI=$AMI"
  ROOT=$(aws ec2 describe-images --image-ids "$AMI" --query 'Images[0].RootDeviceName' --output text)
  echo "root device=$ROOT"

  if [ ! -f "$HOME/.ssh/$KEY.pem" ]; then
    say "key pair $KEY"
    aws ec2 create-key-pair --key-name "$KEY" --query KeyMaterial --output text >"$HOME/.ssh/$KEY.pem"
    chmod 400 "$HOME/.ssh/$KEY.pem"
  fi
  SG=$(aws ec2 describe-security-groups --group-names "$SG_NAME" --query 'SecurityGroups[0].GroupId' \
    --output text 2>/dev/null || true)
  if [ -z "$SG" ] || [ "$SG" = None ]; then
    say "security group $SG_NAME (ssh from this IP only)"
    SG=$(aws ec2 create-security-group --group-name "$SG_NAME" \
      --description "aleph P6 multi-GPU, ssh from my ip" --query GroupId --output text)
    aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 \
      --cidr "$(curl -s https://checkip.amazonaws.com)/32"
  fi

  say "run-instances $TYPE (terminate on shutdown, self-shutdown +240 min, root deleted on termination)"
  IID=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY" \
    --security-group-ids "$SG" \
    --block-device-mappings "[{\"DeviceName\":\"$ROOT\",\"Ebs\":{\"VolumeSize\":200,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
    --instance-initiated-shutdown-behavior terminate \
    --user-data $'#!/bin/bash\nshutdown -h +240\n' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=aleph-p6}]' \
    --query 'Instances[0].InstanceId' --output text)
  aws ec2 wait instance-running --instance-ids "$IID"
  IP=$(aws ec2 describe-instances --instance-ids "$IID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
  printf 'IID=%s\nIP=%s\n' "$IID" "$IP" >"$STATE"
  echo "IID=$IID IP=$IP (saved to $STATE)"
}

setup() {
  load
  say "wait for ssh"
  until rsh true 2>/dev/null; do sleep 10; done
  say "toolchain + NCCL 2.32 (cudarc's nccl-02030 bindings; libnccl-dev gives the unversioned libnccl.so cudarc dlopens)"
  rsh 'set -e; nvidia-smi -L; nvidia-smi topo -m
       command -v cargo || curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
       sudo apt-get update -q
       sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q "libnccl2=2.32.3-1+cuda13.4" "libnccl-dev=2.32.3-1+cuda13.4" \
         || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q libnccl2 libnccl-dev
       ls -la /usr/lib/x86_64-linux-gnu/libnccl.so'
  say "rsync tree"
  rsync -a --delete -e "ssh ${SSH_OPTS[*]}" --exclude target --exclude .git --exclude .superpowers \
    ./ "ubuntu@$IP:aleph/"
}

run() {
  load
  mkdir -p results/p6-aws
  say "topology + oracles (FP64 1e-10 / FP32 1e-5, D=2 and D=4 on real GPUs)"
  rsh 'set -o pipefail; source ~/.cargo/env; cd aleph
       nvidia-smi topo -m > /tmp/topo.txt
       cargo test -p aleph-cuda --features nccl --test dist_nccl_oracle --test dist_gpu_oracle -- --nocapture 2>&1 | tee /tmp/oracle.log' \
    | tail -20
  say "scaling bench + paged baseline"
  rsh 'set -o pipefail; source ~/.cargo/env; cd aleph
       cargo test --release -p aleph-cuda --features nccl --test dist_nccl_bench -- --ignored --nocapture --test-threads 1 2>&1 | tee /tmp/bench.log' \
    | tail -60
  scp "${SSH_OPTS[@]}" "ubuntu@$IP:/tmp/{topo.txt,oracle.log,bench.log}" results/p6-aws/
  echo "results in results/p6-aws/"
}

teardown() {
  load
  say "terminate $IID"
  aws ec2 terminate-instances --instance-ids "$IID" --output text
  aws ec2 wait instance-terminated --instance-ids "$IID"
  say "all-region scan: every line below must be empty"
  for r in $(aws ec2 describe-regions --query 'Regions[].RegionName' --output text); do
    left=$(aws ec2 describe-instances --region "$r" \
      --filters Name=instance-state-name,Values=pending,running,stopping,stopped \
      --query 'Reservations[].Instances[].InstanceId' --output text)
    vols=$(aws ec2 describe-volumes --region "$r" --filters Name=status,Values=available \
      --query 'Volumes[].VolumeId' --output text)
    [ -n "$left$vols" ] && echo "$r: instances=[$left] volumes=[$vols]"
  done
  echo "scan done"
  rm -f "$STATE"
}

case "${1:-}" in
  launch | setup | run | teardown) "$1" ;;
  *) echo "usage: $0 launch|setup|run|teardown" >&2; exit 2 ;;
esac
