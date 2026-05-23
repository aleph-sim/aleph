#!/usr/bin/env bash
# Create the GitHub labels listed in BACKLOG.md § Label System.
# Idempotent: `--force` updates color/description if the label already exists.
set -euo pipefail

# Area labels (purple-ish)
gh label create "area:core"         --color "BFD4F2" --description "Core types and primitives" --force
gh label create "area:parser"       --color "BFD4F2" --description "OpenQASM and other parsers" --force
gh label create "area:ir"           --color "BFD4F2" --description "Circuit IR and optimization passes" --force
gh label create "area:backend"      --color "BFD4F2" --description "Backend trait and dispatch" --force
gh label create "area:backend-sv"   --color "BFD4F2" --description "State vector backend" --force
gh label create "area:backend-mps"  --color "BFD4F2" --description "MPS tensor network backend" --force
gh label create "area:backend-stab" --color "BFD4F2" --description "Stabilizer backend" --force
gh label create "area:backend-gpu"  --color "BFD4F2" --description "GPU acceleration" --force
gh label create "area:backend-dist" --color "BFD4F2" --description "Distributed / multi-GPU" --force
gh label create "area:bench"        --color "BFD4F2" --description "Benchmarking" --force
gh label create "area:infra"        --color "BFD4F2" --description "Build, CI, tooling" --force
gh label create "area:docs"         --color "BFD4F2" --description "Documentation" --force
gh label create "area:python"       --color "BFD4F2" --description "Python bindings" --force
gh label create "area:cli"          --color "BFD4F2" --description "CLI tool" --force

# Type labels
gh label create "type:feature"      --color "1D76DB" --description "New feature or capability" --force
gh label create "type:optimization" --color "0E8A16" --description "Performance optimization" --force
gh label create "type:bug"          --color "D73A4A" --description "Defect fix" --force
gh label create "type:refactor"     --color "FBCA04" --description "Code cleanup, no behavior change" --force
gh label create "type:test"         --color "5319E7" --description "Tests or test infrastructure" --force
gh label create "type:docs"         --color "0075CA" --description "Documentation changes" --force
gh label create "type:infra"        --color "C5DEF5" --description "Build / CI / tooling" --force

# Priority labels
gh label create "priority:critical" --color "B60205" --description "Blocking; must address now" --force
gh label create "priority:high"     --color "D93F0B" --description "Important; address soon" --force
gh label create "priority:medium"   --color "FBCA04" --description "Address when possible" --force
gh label create "priority:low"      --color "C2E0C6" --description "Nice to have" --force

# Difficulty / community labels
gh label create "good-first-issue"  --color "7057FF" --description "Good for newcomers" --force
gh label create "help-wanted"       --color "008672" --description "Help from community welcome" --force
gh label create "research"          --color "FF6B6B" --description "Open-ended; novel territory" --force

echo "All labels created."
