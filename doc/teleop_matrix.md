# Teleop Matrix (2 x 2 x 2)

This matrix defines the first-stage unified router mapping used by `teleop.sh` -> `deploy_real/entrypoints/teleop_entry.py`.

Defaults:

- `policy=twist2`
- `body_source=vdmocap`
- `hand_source=vdhand`

---

## 1) 8-combo mapping

| policy | body_source | hand_source | route type | current target script(s) |
|---|---|---|---|---|
| twist2 | vdmocap | vdhand | single | `xdmocap_teleop.sh` |
| twist2 | vdmocap | manus | dual | `xdmocap_teleop.sh` + `manus_teleop_hand.sh` |
| twist2 | slimevr | vdhand | single | `slimevr_teleop.sh` |
| twist2 | slimevr | manus | dual | `slimevr_teleop.sh` + `manus_teleop_hand.sh` |
| sonic | vdmocap | vdhand | single | `xdmocap_teleop_sonic_v1_mhand.sh` |
| sonic | vdmocap | manus | single | `xdmocap_teleop_sonic_v1_manus.sh` |
| sonic | slimevr | vdhand | single | `slimevr_teleop_sonic_v1_mhand.sh` |
| sonic | slimevr | manus | single | `slimevr_teleop_sonic_v1_manus.sh` |

---

## 2) Notes for v1 router

- v1 focuses on:
  - argument parsing
  - selecting unified pipeline components
  - printing pipeline skeleton (dry-run)
- `teleop.sh` is a thin shell wrapper only.
- Routing logic lives in unified python entrypoint:
  - `deploy_real/entrypoints/teleop_entry.py`
- In this stage, `--run` is intentionally blocked until stage migration is implemented in:
  - `deploy_real/pipelines/body/twist2_pipeline.py`
  - `deploy_real/pipelines/body/sonic_pipeline.py`

---

## 3) Contract with architecture docs

- Terminology: see `doc/ARCH_TERMINOLOGY.md`
- CLI rules: see `doc/ARCH_CLI_SPEC.md`
- Target code layout: see `doc/ARCH_REPO_LAYOUT.md`

