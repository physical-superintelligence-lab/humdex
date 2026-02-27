# Architecture Terminology (Draft v1)

This file freezes naming and terms for the TWIST2_github cleanup.
Scope: repository-level naming only, no runtime behavior changes.

## 1) Canonical Terms

- `policy`
  - `twist2`
  - `sonic`
- `body_source`
  - `vdmocap`
  - `slimevr`
- `hand_source`
  - `vdhand`
  - `manus`
- `target`
  - `sim`
  - `real`

Default runtime selection:

- `policy=twist2`
- `body_source=vdmocap`
- `hand_source=vdhand`
- `target=sim`

## 2) Naming Migration

No compatibility alias is kept for body source naming.

- Only `vdmocap` is valid.
- `xdmocap` is treated as invalid input and should fail fast.

## 3) Teleop Combination Matrix (2 x 2 x 2)

- `policy x body_source x hand_source` = 8 combinations:
  1. `twist2 + vdmocap + vdhand`
  2. `twist2 + vdmocap + manus`
  3. `twist2 + slimevr + vdhand`
  4. `twist2 + slimevr + manus`
  5. `sonic + vdmocap + vdhand`
  6. `sonic + vdmocap + manus`
  7. `sonic + slimevr + vdhand`
  8. `sonic + slimevr + manus`

## 4) Four Product Blocks

- `installation`
  - teleop uses `gmr` env
  - other modules use `humdex` env
- `teleop`
  - one unified entry script
  - selects `policy/body_source/hand_source` from CLI
- `body backend`
  - `twist2`: `sim2sim.sh`, `sim2real.sh`
  - `sonic`: GR00T `deploy.sh`
- `wuji hand backend`
  - `wuji_real.sh`
  - `wuji_sim.sh`

## 5) Style Notes

- Use snake_case in scripts and Python files.
- Keep user-facing names stable once released.

