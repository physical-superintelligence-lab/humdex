#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

usage() {
  cat <<'EOF'
Unified Teleop Entry (v2)

Usage:
  bash teleop.sh [options] [-- extra_args_for_target_scripts]

Options:
  --policy {twist2|sonic}            default: twist2
  --body_source {vdmocap|slimevr}    default: vdmocap
  --hand_source {vdhand|manus}       default: vdhand

  # aliases
  --body {vdmocap|slimevr}
  --hand {vdhand|manus}

  --dry-run                          default behavior (print only)
  --run                              execute resolved command(s)
  -h, --help

Examples:
  bash teleop.sh
  bash teleop.sh --policy sonic --body slimevr --hand manus
  bash teleop.sh --policy sonic --body_source vdmocap --hand_source vdhand --run -- --redis_ip localhost
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cd "${REPO_ROOT}"
exec python -m deploy_real.entrypoints.teleop_entry "$@"
