# Repository Layout Plan (Draft v1)

This document defines the target organization for TWIST2_github cleanup.
It is a plan, not an immediate code move.

## 1) Design Goals

- Keep existing behavior stable.
- Reduce duplication and ultra-long files.
- Separate adapters, pipelines, and publishers.
- Make docs and entrypoints easy to discover.

## 2) Target Structure

```text
TWIST2_github/
  README.md
  teleop.sh
  doc/
    installation.md
    teleop.md
    body_backend.md
    wuji_backend.md
    ARCH_TERMINOLOGY.md
    ARCH_CLI_SPEC.md
    ARCH_REPO_LAYOUT.md

  deploy_real/
    entrypoints/
      teleop_entry.py

    adapters/
      body/
        vdmocap_adapter.py
        slimevr_adapter.py
      hand/
        vdhand_adapter.py
        manus_adapter.py

    pipelines/
      body/
        twist2_pipeline.py
        sonic_pipeline.py
      hand/
        wuji_hand_pipeline.py

    publishers/
      redis_pub.py
      zmq_pub.py

    common/
      cli.py
      keyboard_toggle.py
      rate_limiter.py
      safety_pose.py
      mappings.py

  # existing body backends keep separated
  sim2sim.sh
  sim2real.sh

  # existing wuji hand backends keep separated
  wuji_real.sh
  wuji_sim.sh
```

## 3) Migration Strategy

Phase A (non-breaking):

- Add new folders and shared modules.
- Keep current scripts as-is.
- Root `teleop.sh` is the unified external entry.

Phase B (incremental extraction):

- Move duplicated code from long files into `common/`, `adapters/`, `publishers/`.
- Keep old script names as thin wrappers.

Phase C (default switch):

- Make unified entry the recommended path in README.
- Mark old direct scripts as legacy but still usable.

## 4) Length and Responsibility Rules

- Runner scripts:
  - target < 300 lines
  - only parse config and connect modules
- Adapters:
  - target < 400 lines per source
  - source-specific decoding only
- Pipelines:
  - pure transform logic
  - no direct process/terminal side-effects
- Publishers:
  - transport-only (Redis/ZMQ)
  - no retarget/math logic

## 5) File Naming Rules

- Use snake_case for files and functions.
- Name by responsibility, not by history:
  - preferred: `slimevr_adapter.py`
  - avoid: `*_v1_new2_final.py`

## 6) Entry Rules

- Unified teleop shell entry is at repository root: `teleop.sh`.
- Body backend scripts remain explicit and separate:
  - `sim2sim.sh`
  - `sim2real.sh`
- Wuji backend scripts remain explicit and separate:
  - `wuji_real.sh`
  - `wuji_sim.sh`

## 7) Compatibility Rules

- Keep current script entry names during migration.
- Add deprecation notes only after unified entry is stable.
- Preserve existing key names and message formats unless explicitly versioned.

