from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


def _safe_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / n).astype(np.float32)


SAFE_IDLE_BODY_35_PRESETS: dict[int, list[float]] = {
    0: [
        0.0, 0.0, 0.79, 0.004581602464116093, 0.054385222258041876, -0.01047197449952364, -0.1705406904220581,
        -0.011608824133872986, -0.08608310669660568, 0.2819371521472931, -0.13509835302829742, 0.028368590399622917,
        -0.15945219993591309, -0.011438383720815182, 0.09397093206644058, 0.2500985264778137, -0.12299267947673798,
        0.033810943365097046, 0.01984678953886032, 0.04372693970799446, 0.04439987987279892, -0.052922338247299194,
        0.3638530671596527, 0.018935075029730797, 1.2066316604614258, 0.0026964505668729544, -0.0038426220417022705,
        -0.05543806776404381, 0.016382435336709023, -0.3776109516620636, -0.07517704367637634, 1.2037315368652344,
        -0.03580886498093605, -0.07851681113243103, -0.011213400401175022,
    ],
    1: [
        -2.962986573041272e-06, 6.836185035045111e-06, 0.7900107971067252, 0.026266981563484476, -0.07011304233181229,
        -0.00038564063739400495, 0.21007653006396093, 0.1255744557454361, 0.5210019779740723, -0.087267,
        0.023696508296266388, -0.12259741578159437, 0.18640974335249333, -0.1213838414703421, 0.11017991599235927,
        -0.087267, -0.06074348170695354, 0.10802879748679631, -0.14690420989255235, -0.06195140749854128,
        0.03492134295105836, -0.012934516116481467, 0.012973065503571952, -0.09877424821663634, 1.5735338678105346,
        -0.08846852951921763, -0.008568943127155513, -0.07037145190015832, -0.45191594425028536, -0.7548272891300677,
        0.07631181877180071, 0.623873998918081, 0.32440260037889024, -0.17081521970550126, 0.2697219398563502,
    ],
    2: [
        -0.002425597382764927, 0.0004014222794810171, 0.789948578249186, -0.05286645234860116, -0.11395774381848182,
        -0.0020091780029543797, 0.33550286925644013, 0.07678254800339449, -0.11831599235723278, -0.087267,
        -0.1536621162766681, -0.039016535005063684, 0.28263936593666483, -0.01999487086573224, -0.3918089438082317,
        -0.08726699999999998, -0.06775504509688593, 0.0727761475591654, -0.09677870600760852, -0.0027568505266116657,
        0.07348304585982098, -0.10334908779279858, 0.3160389030446376, 0.07844298473038674, 1.3008225711954524,
        0.6130673022421114, -0.2198179601159421, 0.3438907117467236, -0.23448010297908417, -0.5483439694277361,
        -0.3146753829836872, 0.910606700768848, -0.22716316478096404, -0.10501071874258898, -0.2864687400817216,
    ],
}

# Default body skeleton used by VDMocap SDK initialization.
# Keep this aligned with legacy teleop scripts to avoid constant lower-body bias.
INITIAL_POSITION_BODY = [
    [0, 0, 1.022],
    [0.074, 0, 1.002],
    [0.097, 0, 0.593],
    [0.104, 0, 0.111],
    [0.114, 0.159, 0.005],
    [-0.074, 0, 1.002],
    [-0.097, 0.001, 0.593],
    [-0.104, 0, 0.111],
    [-0.114, 0.158, 0.004],
    [0, 0.033, 1.123],
    [0, 0.03, 1.246],
    [0, 0.014, 1.362],
    [0, -0.048, 1.475],
    [0, -0.048, 1.549],
    [0, -0.016, 1.682],
    [0.071, -0.061, 1.526],
    [0.178, -0.061, 1.526],
    [0.421, -0.061, 1.526],
    [0.682, -0.061, 1.526],
    [-0.071, -0.061, 1.526],
    [-0.178, -0.061, 1.526],
    [-0.421, -0.061, 1.526],
    [-0.682, -0.061, 1.526],
]


def _parse_safe_idle_pose_ids(arg: Any) -> list[int]:
    if arg is None:
        return [0]
    if isinstance(arg, int):
        ids = [int(arg)]
    else:
        s = str(arg).strip()
        if s == "":
            ids = [0]
        else:
            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            ids = [int(p) for p in parts] if parts else [0]
    missing = [i for i in ids if i not in SAFE_IDLE_BODY_35_PRESETS]
    if missing:
        raise ValueError(f"safe_idle_pose_id contains invalid preset ids: {missing}")
    return ids


class SmoothFilter:
    def __init__(self, enable: bool, window_size: int = 5) -> None:
        self.enable = bool(enable)
        self.window_size = max(1, int(window_size))
        self._buf: list[np.ndarray] = []

    def reset(self) -> None:
        self._buf = []

    def apply(self, x: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if x is None:
            return None
        if not self.enable:
            return x
        v = np.asarray(x, dtype=float).copy()
        self._buf.append(v)
        if len(self._buf) > self.window_size:
            self._buf.pop(0)
        if len(self._buf) >= 2:
            return np.mean(np.stack(self._buf, axis=0), axis=0)
        return v


def extract_mimic_obs_whole_body(qpos: np.ndarray, last_qpos: np.ndarray, dt: float) -> np.ndarray:
    from data_utils.rot_utils import euler_from_quaternion_np, quat_diff_np, quat_rotate_inverse_np

    root_pos, last_root_pos = qpos[0:3], last_qpos[0:3]
    root_quat, last_root_quat = qpos[3:7], last_qpos[3:7]
    robot_joints = qpos[7:].copy()
    base_vel = (root_pos - last_root_pos) / dt
    base_ang_vel = quat_diff_np(last_root_quat, root_quat, scalar_first=True) / dt
    roll, pitch, _yaw = euler_from_quaternion_np(root_quat.reshape(1, -1), scalar_first=True)
    base_vel_local = quat_rotate_inverse_np(root_quat, base_vel, scalar_first=True)
    base_ang_vel_local = quat_rotate_inverse_np(root_quat, base_ang_vel, scalar_first=True)
    return np.concatenate([base_vel_local[:2], root_pos[2:3], roll, pitch, base_ang_vel_local[2:3], robot_joints])


INITIAL_POSITION_HAND_RIGHT = [
    [0.682, -0.061, 1.526], [0.71, -0.024, 1.526], [0.728, -0.008, 1.526], [0.755, 0.013, 1.526],
    [0.707, -0.05, 1.526], [0.761, -0.024, 1.525], [0.812, -0.023, 1.525], [0.837, -0.022, 1.525],
    [0.709, -0.058, 1.526], [0.764, -0.046, 1.528], [0.816, -0.046, 1.528], [0.845, -0.046, 1.528],
    [0.709, -0.064, 1.526], [0.761, -0.069, 1.527], [0.812, -0.069, 1.527], [0.835, -0.069, 1.527],
    [0.708, -0.072, 1.526], [0.755, -0.089, 1.522], [0.791, -0.089, 1.522], [0.81, -0.089, 1.522],
]

INITIAL_POSITION_HAND_LEFT = [
    [-0.682, -0.061, 1.526], [-0.71, -0.024, 1.526], [-0.728, -0.008, 1.526], [-0.755, 0.013, 1.526],
    [-0.707, -0.05, 1.526], [-0.761, -0.024, 1.525], [-0.812, -0.023, 1.525], [-0.837, -0.022, 1.525],
    [-0.709, -0.058, 1.526], [-0.764, -0.046, 1.528], [-0.816, -0.046, 1.528], [-0.845, -0.046, 1.528],
    [-0.709, -0.064, 1.526], [-0.761, -0.069, 1.527], [-0.812, -0.069, 1.527], [-0.835, -0.069, 1.527],
    [-0.708, -0.072, 1.526], [-0.755, -0.089, 1.522], [-0.791, -0.089, 1.522], [-0.81, -0.089, 1.522],
]


def _make_tracking_joint(pos: np.ndarray) -> list:
    p = np.asarray(pos, dtype=np.float32).reshape(3)
    return [p.reshape(-1).tolist(), [1.0, 0.0, 0.0, 0.0]]


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = _safe_quat_wxyz(q)
    qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [qw * qw + qx * qx - qy * qy - qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qw * qy + 2 * qx * qz],
            [2 * qw * qz + 2 * qx * qy, qw * qw - qx * qx + qy * qy - qz * qz, 2 * qy * qz - 2 * qw * qx],
            [2 * qx * qz - 2 * qw * qy, 2 * qw * qx + 2 * qy * qz, qw * qw - qx * qx - qy * qy + qz * qz],
        ],
        dtype=np.float32,
    )


def _hand_joint_order_names(prefix: str) -> list[str]:
    p = str(prefix)
    return [
        f"{p}Hand", f"{p}ThumbFinger", f"{p}ThumbFinger1", f"{p}ThumbFinger2",
        f"{p}IndexFinger", f"{p}IndexFinger1", f"{p}IndexFinger2", f"{p}IndexFinger3",
        f"{p}MiddleFinger", f"{p}MiddleFinger1", f"{p}MiddleFinger2", f"{p}MiddleFinger3",
        f"{p}RingFinger", f"{p}RingFinger1", f"{p}RingFinger2", f"{p}RingFinger3",
        f"{p}PinkyFinger", f"{p}PinkyFinger1", f"{p}PinkyFinger2", f"{p}PinkyFinger3",
    ]


def _fk_hand_positions_with_end_sites(
    quats_wxyz: np.ndarray,
    *,
    root_pos: np.ndarray,
    bone_init_pos: np.ndarray,
    end_site_scale: Union[float, np.ndarray] = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(quats_wxyz, dtype=np.float32).reshape(20, 4)
    root_pos = np.asarray(root_pos, dtype=np.float32).reshape(3)
    bone = np.asarray(bone_init_pos, dtype=np.float32).reshape(20, 3)
    parent = np.array([0, 0, 1, 2, 0, 4, 5, 6, 0, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18], dtype=np.int32)
    bone_ver = bone - bone[parent]
    pos = np.zeros((20, 3), dtype=np.float32)
    pos[0] = root_pos
    for j in range(1, 20):
        pidx = int(parent[j])
        R = _quat_wxyz_to_rotmat(q[pidx])
        pos[j] = pos[pidx] + (R @ bone_ver[j].reshape(3, 1)).reshape(3)
    end_joint = np.array([3, 7, 11, 15, 19], dtype=np.int32)
    prev_joint = np.array([2, 6, 10, 14, 18], dtype=np.int32)
    if isinstance(end_site_scale, (list, tuple, np.ndarray)):
        s = np.asarray(end_site_scale, dtype=np.float32).reshape(-1)
        if s.size != 5:
            raise ValueError(f"end_site_scale must be scalar or 5-vector, got shape={s.shape}")
        bone_ver_end = (bone[end_joint] - bone[prev_joint]) * s.reshape(5, 1)
    else:
        bone_ver_end = (bone[end_joint] - bone[prev_joint]) * float(end_site_scale)
    pos_end = np.zeros((5, 3), dtype=np.float32)
    for k in range(5):
        j = int(end_joint[k])
        R = _quat_wxyz_to_rotmat(q[j])
        pos_end[k] = pos[j] + (R @ bone_ver_end[k].reshape(3, 1)).reshape(3)
    return pos, pos_end


def _parse_hand_fk_end_site_scale(v: str) -> Union[float, np.ndarray]:
    s = str(v).strip()
    if "," not in s:
        return float(s)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 5:
        raise ValueError("hand_fk_end_site_scale expects 1 value or 5 comma-separated values")
    return np.asarray([float(x) for x in parts], dtype=np.float32)


def _hand_to_tracking26(
    frame: Dict[str, Any],
    side: str,
    *,
    pos_override: Optional[Dict[str, np.ndarray]] = None,
    tip_override: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    side = side.lower()
    assert side in ["left", "right"]
    pfx = "Left" if side == "left" else "Right"
    out: Dict[str, Any] = {}

    def get_pos(name: str) -> np.ndarray:
        if pos_override is not None and name in pos_override:
            return np.asarray(pos_override[name], dtype=np.float32).reshape(3)
        v = frame.get(name, None)
        if isinstance(v, (list, tuple)) and len(v) >= 1:
            return np.asarray(v[0], dtype=np.float32).reshape(3)
        return np.zeros(3, dtype=np.float32)

    def pick(*names: str) -> str:
        for n in names:
            if n in frame:
                return n
        return names[0]

    WUJI_FINGER_TIP_SCALING = [1.0, 1.0, 1.0, 1.0, 1.0]

    def scale_chain_4seg(o: np.ndarray, m: np.ndarray, p: np.ndarray, i: np.ndarray, d: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        m2 = o + s * (m - o)
        p2 = m2 + s * (p - m)
        i2 = p2 + s * (i - p)
        d2 = i2 + s * (d - i)
        return m2, p2, i2, d2

    def scale_chain_3seg(o: np.ndarray, m: np.ndarray, p: np.ndarray, d: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m2 = o + s * (m - o)
        p2 = m2 + s * (p - m)
        d2 = p2 + s * (d - p)
        return m2, p2, d2

    hand_pos = get_pos(pick(f"{pfx}Hand"))
    out[f"{pfx}HandWrist"] = _make_tracking_joint(hand_pos)
    out[f"{pfx}HandPalm"] = _make_tracking_joint(hand_pos)

    th_m2, th_p2, th_d2 = scale_chain_3seg(
        hand_pos,
        get_pos(pick(f"{pfx}ThumbFinger")),
        get_pos(pick(f"{pfx}ThumbFinger1", f"{pfx}ThumbFinger")),
        get_pos(pick(f"{pfx}ThumbFinger2", f"{pfx}ThumbFinger1")),
        float(WUJI_FINGER_TIP_SCALING[0]),
    )
    out[f"{pfx}HandThumbMetacarpal"] = _make_tracking_joint(th_m2)
    out[f"{pfx}HandThumbProximal"] = _make_tracking_joint(th_p2)
    out[f"{pfx}HandThumbDistal"] = _make_tracking_joint(th_d2)
    out[f"{pfx}HandThumbTip"] = _make_tracking_joint(tip_override[f"{pfx}HandThumbTip"]) if (tip_override is not None and f"{pfx}HandThumbTip" in tip_override) else _make_tracking_joint(th_d2)

    def fill_finger(base: str, scale_idx: int, mk: str, pk: str, ik: str, dk: str, tip_key: str) -> None:
        m2, p2, i2, d2 = scale_chain_4seg(hand_pos, get_pos(pick(mk)), get_pos(pick(pk, mk)), get_pos(pick(ik, pk)), get_pos(pick(dk, ik)), float(WUJI_FINGER_TIP_SCALING[scale_idx]))
        out[f"{pfx}Hand{base}Metacarpal"] = _make_tracking_joint(m2)
        out[f"{pfx}Hand{base}Proximal"] = _make_tracking_joint(p2)
        out[f"{pfx}Hand{base}Intermediate"] = _make_tracking_joint(i2)
        out[f"{pfx}Hand{base}Distal"] = _make_tracking_joint(d2)
        if tip_override is not None and tip_key in tip_override:
            out[tip_key] = _make_tracking_joint(tip_override[tip_key])
        else:
            out[tip_key] = _make_tracking_joint(d2)

    fill_finger("Index", 1, f"{pfx}IndexFinger", f"{pfx}IndexFinger1", f"{pfx}IndexFinger2", f"{pfx}IndexFinger3", f"{pfx}HandIndexTip")
    fill_finger("Middle", 2, f"{pfx}MiddleFinger", f"{pfx}MiddleFinger1", f"{pfx}MiddleFinger2", f"{pfx}MiddleFinger3", f"{pfx}HandMiddleTip")
    fill_finger("Ring", 3, f"{pfx}RingFinger", f"{pfx}RingFinger1", f"{pfx}RingFinger2", f"{pfx}RingFinger3", f"{pfx}HandRingTip")
    fill_finger("Little", 4, f"{pfx}PinkyFinger", f"{pfx}PinkyFinger1", f"{pfx}PinkyFinger2", f"{pfx}PinkyFinger3", f"{pfx}HandLittleTip")
    return out


__all__ = [
    "SAFE_IDLE_BODY_35_PRESETS",
    "INITIAL_POSITION_BODY",
    "INITIAL_POSITION_HAND_LEFT",
    "INITIAL_POSITION_HAND_RIGHT",
    "SmoothFilter",
    "_fk_hand_positions_with_end_sites",
    "_hand_joint_order_names",
    "_hand_to_tracking26",
    "_parse_hand_fk_end_site_scale",
    "_parse_safe_idle_pose_ids",
    "_safe_quat_wxyz",
    "extract_mimic_obs_whole_body",
]
