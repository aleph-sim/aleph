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
#   scripts/aws/p6-multi-gpu-session.sh setup      # rustup + NCCL 2.32 + NVRTC 13 + preflight + rsync
#   scripts/aws/p6-multi-gpu-session.sh run        # oracles, bench, paged baseline → results/p6-aws/
#   scripts/aws/p6-multi-gpu-session.sh teardown   # terminate + all-region scan (works without state)
# State (IID, IP) and logs live in results/p6-aws/ (git-ignored).
set -euo pipefail

REGION=us-east-1
TYPE=g6.12xlarge
KEY=aleph-p6
SG_NAME=aleph-p6
OUT=results/p6-aws
STATE=$OUT/.state
SSH_OPTS=(-i "$HOME/.ssh/$KEY.pem" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
# Remote env for every test run: fail (not skip) with fewer than 4 GPUs, and
# keep NCCL's own diagnostics (nccl_err drops the result code). g6.12xlarge
# has no GPU P2P (`nvidia-smi topo -p2p r` = NS), so NCCL stages through host
# SHM; its default SM-driven copy gets ~2.9 GB/s/GPU, the copy-engine path
# (SHM_USE_CUDA_MEMCPY) ~7.2 GB/s (measured 2026-10-04, docs/perf/p6-multi-gpu.md §4).
REMOTE_ENV='source ~/.cargo/env; cd aleph; export ALEPH_REQUIRE_GPUS=4 NCCL_DEBUG=WARN NCCL_SHM_USE_CUDA_MEMCPY=1'
export AWS_DEFAULT_REGION=$REGION
mkdir -p "$OUT"

say() { printf '\n== %s\n' "$*"; }
load() {
  # shellcheck source=/dev/null
  [ -f "$STATE" ] && . "$STATE"
  : "${IID:?no IID in $STATE — run launch first}"
  : "${IP:?no IP in $STATE}"
}
rsh() { ssh "${SSH_OPTS[@]}" "ubuntu@$IP" "$@"; }

launch() {
  if [ -f "$STATE" ]; then
    echo "$STATE exists (a session may be live): run teardown first" >&2
    exit 1
  fi
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
    say "security group $SG_NAME"
    SG=$(aws ec2 create-security-group --group-name "$SG_NAME" \
      --description "aleph P6 multi-GPU, ssh from my ip" --query GroupId --output text)
  fi
  # Idempotent: (re)open ssh for *this* machine's current IP on every launch.
  MYIP=$(curl -s https://checkip.amazonaws.com)
  aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 \
    --cidr "$MYIP/32" 2>/dev/null || echo "ingress for $MYIP/32 already present"

  say "run-instances $TYPE (terminate on shutdown, self-shutdown +240 min, root deleted on termination)"
  IID=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY" \
    --security-group-ids "$SG" \
    --block-device-mappings "[{\"DeviceName\":\"$ROOT\",\"Ebs\":{\"VolumeSize\":200,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
    --instance-initiated-shutdown-behavior terminate \
    --user-data $'#!/bin/bash\nshutdown -h +240\n' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=aleph-p6}]' \
    --query 'Instances[0].InstanceId' --output text)
  # Record the IID before anything else can fail, so teardown always finds it.
  printf 'IID=%s\nIP=\n' "$IID" >"$STATE"
  echo "IID=$IID (saved)"
  aws ec2 wait instance-running --instance-ids "$IID"
  IP=$(aws ec2 describe-instances --instance-ids "$IID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
  printf 'IID=%s\nIP=%s\n' "$IID" "$IP" >"$STATE"
  echo "IID=$IID IP=$IP"
}

setup() {
  load
  say "wait for ssh (max 10 min)"
  for i in $(seq 60); do
    rsh true 2>/dev/null && break
    [ "$i" = 60 ] && { echo "ssh never came up: check SG ingress / instance state; then teardown" >&2; exit 1; }
    sleep 10
  done
  say "toolchain, NCCL 2.32 (libnccl-dev = the unversioned libnccl.so cudarc dlopens), CUDA 13 NVRTC"
  rsh 'set -e
       command -v cargo >/dev/null || curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
       if ! apt-cache policy libnccl2 | grep -q "cuda13"; then
         wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
         sudo dpkg -i cuda-keyring_1.1-1_all.deb
       fi
       sudo apt-get update -q
       sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
         "libnccl2=2.32.3-1+cuda13.4" "libnccl-dev=2.32.3-1+cuda13.4" cuda-nvrtc-13-0'
  say "preflight: driver >= 580 (cuda-13000 bindings), libcuda / libnvrtc / libnccl.so loadable, 4 GPUs"
  rsh 'set -e
       nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
       drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
       [ "$drv" -ge 580 ] || { echo "driver $drv < 580: CUDA 13 bindings will not load" >&2; exit 1; }
       [ "$(nvidia-smi -L | wc -l)" -ge 4 ] || { echo "fewer than 4 GPUs" >&2; exit 1; }
       echo /usr/local/cuda-13.0/lib64 | sudo tee /etc/ld.so.conf.d/cuda13.conf >/dev/null; sudo ldconfig
       ldconfig -p | grep -E "libcuda\.so|libnvrtc\.so|libnccl\.so" '
  say "rsync tree"
  rsync -a --delete -e "ssh ${SSH_OPTS[*]}" --exclude target --exclude .git --exclude .superpowers \
    --exclude results ./ "ubuntu@$IP:aleph/"
  say "smoke: backends on all 4 GPUs (non-skipping)"
  rsh "$REMOTE_ENV; timeout 20m cargo test -p aleph-cuda --features nccl --test dist_gpu_oracle backends_open -- --test-threads=1 2>&1 | tail -3"
}

run() {
  load
  # Fetch logs whatever happens below (a failing oracle must not strand them).
  trap 'scp "${SSH_OPTS[@]}" "ubuntu@$IP:/tmp/p6-*.log" "ubuntu@$IP:/tmp/p6-topo.txt" "$OUT/" 2>/dev/null || true; echo "logs → $OUT/"' EXIT
  rsh 'nvidia-smi topo -m > /tmp/p6-topo.txt; cat /tmp/p6-topo.txt'
  say "oracles: FP64 1e-10 / FP32 1e-5 at D=2,4 on real GPUs (fail, not skip, under 4 GPUs)"
  rsh "$REMOTE_ENV; timeout 40m cargo test -p aleph-cuda --features nccl \
       --test dist_nccl_oracle --test dist_gpu_oracle --test dist_host_overlap --test dist_nccl_memcpy_shm \
       -- --test-threads=1 --nocapture \
       > /tmp/p6-oracle.log 2>&1; s=\$?; grep -E 'test result|panicked|FAILED' /tmp/p6-oracle.log; exit \$s"
  say "scaling bench (strong n=28, weak m=30/31)"
  rsh "$REMOTE_ENV; timeout 90m cargo test --release -p aleph-cuda --features nccl --test dist_nccl_bench \
       dist_nccl_scaling -- --ignored --nocapture --test-threads=1 > /tmp/p6-bench.log 2>&1; s=\$?; \
       grep -E '^(gpus|xchg|strong|weak)' /tmp/p6-bench.log; exit \$s"
  # On g6.12xlarge (L4 = PCIe Gen4 x8) one FP64 n=31 paged run alone took > 48 min
  # (2026-10-04); n=32/33 take hours. The multi-GPU verdict does not need them.
  say "paged single-GPU weak baseline (very long: run last; skip with Ctrl-C if the clock is short)"
  rsh "$REMOTE_ENV; timeout 80m cargo test --release -p aleph-cuda --features nccl --test dist_nccl_bench \
       weak_paged_baseline -- --ignored --nocapture --test-threads=1 > /tmp/p6-paged.log 2>&1; s=\$?; \
       grep -E '^weak_paged' /tmp/p6-paged.log; exit \$s"
}

teardown() {
  # No `load`: tear down even when the state is partial or missing.
  IID=""
  # shellcheck source=/dev/null
  [ -f "$STATE" ] && . "$STATE"
  if [ -n "${IID:-}" ]; then
    say "terminate $IID"
    aws ec2 terminate-instances --instance-ids "$IID" --output text
    aws ec2 wait instance-terminated --instance-ids "$IID"
  else
    echo "no IID recorded — relying on the scan below (look for Name=aleph-p6)"
  fi
  say "all-region scan: no line may print between here and 'scan done'"
  for r in $(aws ec2 describe-regions --query 'Regions[].RegionName' --output text); do
    left=$(aws ec2 describe-instances --region "$r" \
      --filters Name=instance-state-name,Values=pending,running,stopping,stopped \
      --query 'Reservations[].Instances[].[InstanceId,Tags[?Key==`Name`]|[0].Value]' --output text)
    vols=$(aws ec2 describe-volumes --region "$r" --filters Name=status,Values=available \
      --query 'Volumes[].VolumeId' --output text)
    if [ -n "$left$vols" ]; then echo "$r: instances=[$left] volumes=[$vols]"; fi
  done
  echo "scan done"
  rm -f "$STATE"
}

case "${1:-}" in
  launch | setup | run | teardown) "$1" ;;
  *) echo "usage: $0 launch|setup|run|teardown" >&2; exit 2 ;;
esac
