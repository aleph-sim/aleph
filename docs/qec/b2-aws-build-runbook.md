# Task B2 Step 1 — AWS build-instance runbook

Everything here is command line except one step that AWS does not expose to the CLI. Follow it in
order; the first section is the one that will otherwise waste your day.

**What you are renting and why:** not an FPGA. Steps 2–3 of Task B2 want utilisation and achieved
Fmax, which is synthesis and implementation — no bitstream is ever loaded. The only thing AWS supplies
that we cannot get elsewhere is a **Vivado licence for a large AMD part**, bundled free in the FPGA
Developer AMI and valid only on EC2. See `docs/perf/q7-02-fullparallel-fpga.md` §6.3.

**Expected cost:** $30–60 for ~40 instance-hours. Set the budget alarm in §1 anyway.

**Expected wall-clock:** quota approval 0–24 h (do it first), then ~1 h of setup, then 4–12 h per
implementation run.

-----

## 0. Prerequisites

An AWS account and the CLI:

```bash
brew install awscli          # macOS
aws configure                # access key, secret, default region us-east-1, output json
aws sts get-caller-identity  # must print your account id
```

Use `us-east-1`. Not because it is better, but because every AMI and instance type appears there first
and the FPGA Developer AMI is definitely published there.

-----

## 1. Do this before anything else: the vCPU quota

**A brand-new AWS account has a limit of 5 vCPUs for standard on-demand instances.** `z1d.2xlarge` is
8 vCPUs. Your launch will fail with `VcpuLimitExceeded` and nothing in the console will have warned
you. Approval takes anywhere from minutes to a day, so start it now and read the rest while waiting.

```bash
# What you have today (quota code L-1216C47A = "Running On-Demand Standard (A,C,D,H,I,M,R,T,Z)")
aws service-quotas get-service-quota \
  --service-code ec2 --quota-code L-1216C47A \
  --query 'Quota.[QuotaName,Value]' --output text

# Ask for 16 — one 8-vCPU box plus headroom to launch a replacement before killing the first
aws service-quotas request-service-quota-increase \
  --service-code ec2 --quota-code L-1216C47A --desired-value 16

# Poll until CASE_CLOSED / APPROVED
aws service-quotas list-requested-service-quota-change-history-by-quota \
  --service-code ec2 --quota-code L-1216C47A \
  --query 'RequestedQuotas[0].[Status,DesiredValue]' --output text
```

While you are here, set a spending guardrail. This is the part of AWS that actually deserves the
distrust:

```bash
aws budgets create-budget --account-id "$(aws sts get-caller-identity --query Account --output text)" \
  --budget '{"BudgetName":"b2-build","BudgetLimit":{"Amount":"100","Unit":"USD"},
             "TimeUnit":"MONTHLY","BudgetType":"COST"}'
```

-----

## 2. The one console step: subscribe to the AMI

AWS Marketplace requires you to accept the EULA in a browser once per account. There is no CLI for it.

1. Open <https://aws.amazon.com/marketplace/pp/prodview-tcl7sjgreh6bq> — **FPGA Developer AMI (Ubuntu)**.
   The Rocky Linux variant is at <https://aws.amazon.com/marketplace/pp/prodview-7mukkbz7l2uvu> and is
   equally fine; pick whichever you would rather debug on.
2. Click **Continue to Subscribe**, then **Accept Terms**.
3. Wait for the subscription to go green. **Do not continue to "Launch"** — that path drops you into
   the console launch wizard. Close the tab; you are done here.

The AMI itself is **$0** — you pay only EC2 and EBS.

> Do not confuse it with the *"Vivado ML Developer AMI"* listings published by AMD. Those are a
> different product with different licensing. You want the one named **FPGA Developer AMI**, which is
> the one carrying the licence for the parts AWS deploys.

-----

## 3. Find the AMI id

Do not hardcode an id from any document, including this one — they are region-specific and are
republished on every tool release.

```bash
aws ec2 describe-images --owners aws-marketplace \
  --filters 'Name=name,Values=FPGA Developer AMI*' \
  --query 'sort_by(Images,&CreationDate)[-3:].[ImageId,Name,CreationDate]' --output table
```

At the time of writing the newest were `ami-0ade775ce1137d8d3` (Rocky Linux 8.10) and
`ami-017bd23ff95264395` (Ubuntu 24.04) in `us-east-1`, both carrying **Vivado 2025.2**.

> **Note the tool version.** Our Step 0 sweep ran Vivado **2024.2** on the EPYC box. Numbers from
> 2025.2 are a fresh measurement on a different part, not a like-for-like comparison with Step 0 — say
> so in the report rather than tabulating them side by side as if they were.

```bash
export AMI=ami-xxxxxxxxxxxx   # whatever the query above returned
```

-----

## 4. Key pair and security group

A security group cannot exist outside a VPC, and not every account has a **default VPC** in every
region any more — if it was deleted, or the account is new or organisation-managed, you get
`VPCIdNotSpecified` here rather than anywhere more informative. Check first:

```bash
aws configure get region     # and check $AWS_DEFAULT_REGION — the env var wins over this
aws ec2 describe-vpcs --query 'Vpcs[].[VpcId,IsDefault,CidrBlock]' --output table
```

If nothing is `True`, recreate the default VPC — one command, and it brings subnets in every AZ, an
internet gateway and routes with it:

```bash
aws ec2 create-default-vpc --query 'Vpc.VpcId' --output text
```

If you must use a non-default VPC instead, pass `--vpc-id` below *and* add `--subnet-id` plus
`--associate-public-ip-address` to `run-instances` — without a default VPC nothing is chosen for you,
and an instance with no public IP is an instance you cannot reach.

```bash
aws ec2 create-key-pair --key-name b2build \
  --query KeyMaterial --output text > ~/.ssh/b2build.pem
chmod 400 ~/.ssh/b2build.pem

export SG=$(aws ec2 create-security-group --group-name b2build \
  --description "B2 build box, ssh from my ip only" --query GroupId --output text)

aws ec2 authorize-security-group-ingress --group-id "$SG" \
  --protocol tcp --port 22 --cidr "$(curl -s https://checkip.amazonaws.com)/32"
```

The `/32` matters. An SSH port open to `0.0.0.0/0` is found by scanners within minutes.

-----

## 5. Launch

Check what the AMI's root device is actually called and how big its snapshot is — both feed the next
command, and getting either wrong fails in a way that is easy to misread:

```bash
aws ec2 describe-images --image-ids "$AMI" \
  --query 'Images[0].[RootDeviceName,BlockDeviceMappings[0].Ebs.VolumeSize]' --output text
```

The FPGA Developer AMI's root snapshot is **120 GB** — Vivado is large — so anything smaller is
rejected with `InvalidBlockDeviceMapping`. Ask for 150 to leave room for checkpoints and reports; gp3
is about $0.08/GB-month, so the extra 30 GB for a couple of days costs cents.

```bash
aws ec2 run-instances \
  --image-id "$AMI" \
  --instance-type z1d.2xlarge \
  --key-name b2build \
  --security-group-ids "$SG" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1",
      "Ebs":{"VolumeSize":150,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=b2build}]' \
  --query 'Instances[0].InstanceId' --output text
```

Two flags earn their place:

- `DeleteOnTermination: true` — otherwise the 150 GB volume outlives the instance and bills quietly
  forever. This is the single most common way to keep paying AWS for something you thought you deleted.
- `--instance-initiated-shutdown-behavior terminate` — lets the build script end with `sudo shutdown -h
  now` and destroy the box by itself. Use it (§7). It is the only reliable defence against forgetting.

Connect:

```bash
export IID=i-xxxxxxxxxxxx
export IP=$(aws ec2 describe-instances --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
ssh -i ~/.ssh/b2build.pem ubuntu@"$IP"      # 'rocky@' on the Rocky Linux AMI
```

`z1d.2xlarge` carries a ~300 GB NVMe instance store. It is free, fast, and wiped on termination —
exactly right for Vivado scratch. Put the run directory there and keep only reports on EBS:

```bash
lsblk                                    # find the NVMe device, typically nvme1n1
sudo mkfs.ext4 /dev/nvme1n1 && sudo mkdir -p /scratch && sudo mount /dev/nvme1n1 /scratch
sudo chown $USER /scratch
```

-----

## 6. Smoke-test the licence before spending hours

**Do this first.** The AMI's licence covers the parts AWS deploys. If `xcvu47p` is not among them you
want to know in five minutes, not after a twelve-hour place-and-route dies at the end.

```bash
source /opt/Xilinx/Vivado/*/settings64.sh     # confirm the path with: ls -d /opt/Xilinx/Vivado/*
cd /scratch && mkdir lic && cd lic
cat > t.v <<'EOF'
module t(input clk, input a, input b, output reg y);
  always @(posedge clk) y <= a & b;
endmodule
EOF
cat > t.tcl <<'EOF'
read_verilog t.v
synth_design -top t -part xcvu47p-fsvh2892-2-e -mode out_of_context
puts "LICENCE_OK [get_property PART [current_design]]"
EOF
vivado -mode batch -source t.tcl 2>&1 | tail -20
```

`LICENCE_OK …` means proceed. A licence error naming the part means the AMI does not cover it — stop,
and re-read §6.3 of the Step 0 report, because the plan changes.

> The exact part suffix (`-fsvh2892-2-e`) is a guess at the package and speed grade. If Vivado rejects
> it, list what it does have: `puts [get_parts -filter {DEVICE =~ *vu47p*}]`, and use one of those.
> Prefer the speed grade F2 actually ships; a `-3` part would flatter the Fmax number.

-----

## 7. Stage and run the build

From your laptop, reusing exactly the staging layout Step 0 used — one directory per geometry holding
`check_minsum.sv`, `var_update.sv`, `bp_relay_banked.sv` and that geometry's generated
`bb_gross_tanner.svh`:

```bash
# generate the two headers (from the repo root)
cargo build --release -q -p aleph-qec --example qec_q7_bp_graph
for wv in "64 192" "144 864"; do set -- $wv          # bash, not zsh — see the gotcha below
  d="stage/${1}x${2}"; mkdir -p "$d"
  ./target/release/examples/qec_q7_bp_graph circgraph 1 0.003 "$1" "$2" > "$d/bb_gross_tanner.svh"
  cp hw/check_minsum.sv hw/var_update.sv hw/bp_relay_banked.sv "$d/"
done
scp -i ~/.ssh/b2build.pem -r stage hw/syn/ooc_banked.tcl ubuntu@"$IP":/scratch/
```

`ooc_banked.tcl` is an **out-of-context synthesis** probe — it stops after `synth_design`, which is
what Step 0 measured. Step 1 exists to go further, so the run script must add `opt_design`,
`place_design`, `route_design` and report **post-route** utilisation and timing. That is the whole
point of renting the box; a second OOC number would tell us nothing we do not already have.

Change the part to `xcvu47p-…` and drive it with something that self-terminates:

```bash
cat > /scratch/run.sh <<'EOF'
#!/bin/bash
set -u
source /opt/Xilinx/Vivado/*/settings64.sh
cd /scratch
for g in 64x192 144x864; do          # smaller first: the fallback config must land even if the big one dies
  ( cd "$g" && vivado -mode batch -source ../impl_vu47p.tcl -tclargs 5.0 "$g" >"impl_$g.log" 2>&1 )
  echo "$g rc=$? $(grep -m1 '^RESULT ' "$g/impl_$g.log")" >>/scratch/summary.txt
done
echo DONE >>/scratch/summary.txt
sudo shutdown -h now                 # terminates the instance; see --instance-initiated-shutdown-behavior
EOF
chmod +x /scratch/run.sh
nohup setsid /scratch/run.sh >/dev/null 2>&1 </dev/null &
```

Detach it. An implementation run outlives any SSH session you will keep open.

**Pull the results before the box terminates itself.** Either drop the `shutdown` line until you have
copied them off, or have the script `aws s3 cp` them out first. Losing a twelve-hour run to its own
cleanup is an avoidable way to pay twice.

```bash
scp -i ~/.ssh/b2build.pem "ubuntu@$IP:/scratch/{summary.txt,*/impl_*.log,*/*.rpt}" ./results/
```

-----

## 8. Shut it down and verify

Even with self-termination, check. AWS bills for what exists, not for what you meant to delete.

```bash
aws ec2 terminate-instances --instance-ids "$IID"

# Nothing should be left running, and no volume should be sitting 'available'
aws ec2 describe-instances --filters 'Name=instance-state-name,Values=running,pending,stopped' \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name]' --output table
aws ec2 describe-volumes --filters 'Name=status,Values=available' \
  --query 'Volumes[].[VolumeId,Size]' --output table
aws ec2 describe-addresses --query 'Addresses[].[PublicIp,InstanceId]' --output table
```

A `stopped` instance still bills for its EBS volume. An unattached Elastic IP bills by the hour. Both
tables should be empty.

-----

## 9. Gotchas, in the order they will bite

1. **`VcpuLimitExceeded` on launch** — §1. New accounts get 5 vCPUs; you need 8.
2. **`VPCIdNotSpecified` on the security group** — no default VPC in this region. §4.
3. **`InvalidBlockDeviceMapping … smaller than snapshot`** — the AMI's root snapshot is 120 GB. Ask for
   150. §5.
4. **A `DeviceName` that does not match the AMI's root device** fails *silently* in the worst way: AWS
   creates a second volume and leaves root at its default size. Confirm it with `describe-images`
   rather than trusting the `/dev/sda1` in this document.
5. **`OptInRequired` on launch** — you skipped §2, or subscribed in a different account than the one
   your CLI credentials belong to.
6. **The EBS volume outliving the instance** — always `DeleteOnTermination: true`.
7. **zsh does not word-split.** The `for wv in "64 192"; set -- $wv` idiom in §7 gives `$1="64 192"`
   and `$2=""` under zsh, and silently generates the *default* geometry. Run that loop under bash. This
   has already cost this project a debugging session.
8. **Vivado version drift.** The AMI ships 2025.2; Step 0 ran 2024.2. Report them as separate
   measurements, never as a before/after.
9. **Speed grade.** A `-3` part will flatter Fmax against a `-2`. Record which one you used, in the
   report, next to the number.
10. **SLR partitioning.** VU47P is a multi-die part. An 800k-LUT monolithic design will be split across
   super logic regions and the crossings are expensive. If post-route Fmax collapses against the OOC
   estimate, this is the first suspect — check the SLR-crossing count before concluding the design is
   slow.
11. **Do not create `hw/syn/f1_144x864.tcl`.** The program document asks for it; F1 is end-of-life and
   VU9P is not rentable. Name it for the part you actually used.
