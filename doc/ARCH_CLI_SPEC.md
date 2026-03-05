# Unified Teleop CLI Spec (Draft v1)

This spec defines the future unified teleop entry contract.
Goal: one clean CLI, stable naming, and deterministic routing.

## 1) Entry

Recommended unified entry:

```bash
bash teleop.sh \
  --policy sonic \
  --body_source slimevr \
  --hand_source manus
```

Default entry behavior (when args are omitted):

- `--policy twist2`
- `--body_source vdmocap`
- `--hand_source vdhand`

## 2) Required Arguments

- `--policy {twist2|sonic}`
- `--body_source {vdmocap|slimevr}`
- `--hand_source {vdhand|manus}`

`xdmocap` is not supported by CLI contract.

## 3) Core Optional Arguments

- `--redis_ip <ip>` (default: `localhost`)
- `--target_fps <float>`
- `--enable_zmq_pose`
- `--zmq_pose_port <int>`
- `--zmq_pose_topic <string>`
- `--keyboard_toggle_send`
- `--toggle_send_key <char>`
- `--hold_position_key <char>`

Source-specific optional args:

- VDMocap:
  - `--vdmocap_ip`
  - `--vdmocap_port`
- SlimeVR:
  - `--vmc_ip`
  - `--vmc_port`
  - `--vmc_use_fk`
  - `--vmc_use_viewer_fk`
  - `--vmc_bvh_path`
  - `--vmc_bvh_scale`
- Manus:
  - `--manus_address`
  - `--manus_left_sn`
  - `--manus_right_sn`
- VDHand:
  - keep existing vdhand-specific options (to be listed in implementation phase)

## 4) Routing Contract (Unified Python Entry)

`teleop.sh` should do only:

1. Parse and normalize args.
2. Pass args to one unified Python entrypoint.
3. Unified Python entrypoint selects modules by args.
4. Print selected modules before execution.

Important design choice:

- Not one-combo-one-python-file.
- Keep one Python entrypoint plus modular utils.
- Concrete module split and naming are defined in `doc/ARCH_REPO_LAYOUT.md`.

## 5) Error Policy

- Unknown enum value -> fail fast with allowed values.
- Unsupported combo -> fail with explicit message.
- Missing source-specific required params -> fail with fix suggestion.

## 6) Backward Compatibility

- Existing script entrypoints may remain valid during migration.
- New architecture does not preserve `xdmocap` naming in unified CLI.

