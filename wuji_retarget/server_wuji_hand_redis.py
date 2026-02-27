#!/usr/bin/env python3
"""
Wuji Hand Controller via Redis

从 Redis 读取 teleop.sh 发送的手部追踪数据（26维），转换为21维 MediaPipe 格式，
然后使用 WujiHandRetargeter 进行重定向，实时控制 Wuji 灵巧手。
"""

import argparse
import json
from re import T
import time
import numpy as np
import redis
import signal
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
from typing import Any

# Optional OpenCV for visualization
try:
    import cv2  # type: ignore
    _CV2_AVAILABLE = True
except Exception:
    cv2 = None
    _CV2_AVAILABLE = False

# Optional Open3D for true 3D visualization
try:
    import open3d as o3d  # type: ignore
    _O3D_AVAILABLE = True
    _O3D_IMPORT_ERROR = None
except Exception as e:
    o3d = None
    _O3D_AVAILABLE = False
    _O3D_IMPORT_ERROR = repr(e)


# MediaPipe 21 keypoints hand skeleton connections (indices)
# 0: wrist
# Thumb: 1-4, Index: 5-8, Middle: 9-12, Ring: 13-16, Pinky: 17-20
_MP_HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17),
]


@dataclass
class _HandVizView:
    """21D 手关键点可视化视角（简单 3D 旋转 + 缩放），用于把 3D 点旋转后投影到 2D。"""
    yaw_deg: float = 0.0    # 绕 Z
    pitch_deg: float = 0.0  # 绕 X
    roll_deg: float = 0.0   # 绕 Y
    scale_px_per_m: float = 1200.0


def _rot_x(a: float) -> np.ndarray:
    ca, sa = float(np.cos(a)), float(np.sin(a))
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=np.float32)


def _rot_y(a: float) -> np.ndarray:
    ca, sa = float(np.cos(a)), float(np.sin(a))
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=np.float32)


def _rot_z(a: float) -> np.ndarray:
    ca, sa = float(np.cos(a)), float(np.sin(a))
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=np.float32)


def _viz_draw_mediapipe_hand_21d(
    pts21: np.ndarray,
    win_name: str = "hand_21d",
    canvas_size: int = 640,
    scale_px_per_m: float = 1200.0,
    flip_y: bool = True,
    show_index: bool = False,
    hand_side: str = "",
    view: Optional[_HandVizView] = None,
) -> Tuple[bool, _HandVizView]:
    """
    简单实时可视化 21D 手关键点（3D 旋转后投影到二维 x-y），带连线。

    - 手腕(0)和五个指尖(4/8/12/16/20)标红
    - 其他点标绿

    Returns:
        (ok, view)
        - ok=True 表示正常显示
        - ok=False 表示无法显示或用户退出（例如无 GUI / cv2 不可用 / 按 q/ESC）
    """
    if view is None:
        view = _HandVizView(scale_px_per_m=float(scale_px_per_m))
    if not _CV2_AVAILABLE or cv2 is None:
        return False, view
    if pts21 is None:
        return False, view
    pts21 = np.asarray(pts21, dtype=np.float32).reshape(21, 3)

    H = int(canvas_size)
    W = int(canvas_size)
    img = np.zeros((H, W, 3), dtype=np.uint8)

    cx = W // 2
    cy = H // 2

    # 3D rotate -> 2D project
    yaw = float(np.deg2rad(view.yaw_deg))
    pitch = float(np.deg2rad(view.pitch_deg))
    roll = float(np.deg2rad(view.roll_deg))
    R = _rot_z(yaw) @ _rot_x(pitch) @ _rot_y(roll)
    pts3 = (R @ pts21.T).T  # (21,3)
    xy = pts3[:, :2].copy()
    if flip_y:
        xy[:, 1] *= -1.0

    pts2 = np.zeros((21, 2), dtype=np.int32)
    s = float(view.scale_px_per_m)
    pts2[:, 0] = (cx + xy[:, 0] * s).astype(np.int32)
    pts2[:, 1] = (cy + xy[:, 1] * s).astype(np.int32)

    # Draw connections
    for a, b in _MP_HAND_CONNECTIONS:
        pa = tuple(int(x) for x in pts2[a])
        pb = tuple(int(x) for x in pts2[b])
        cv2.line(img, pa, pb, (180, 180, 180), 2, lineType=cv2.LINE_AA)

    # Draw points
    tip_idxs = {4, 8, 12, 16, 20}
    for i in range(21):
        p = tuple(int(x) for x in pts2[i])
        is_red = (i == 0) or (i in tip_idxs)
        color = (0, 0, 255) if is_red else (0, 255, 0)  # BGR
        radius = 7 if is_red else 5
        cv2.circle(img, p, radius, color, -1, lineType=cv2.LINE_AA)
        if show_index:
            cv2.putText(img, str(i), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    title = "21D MediaPipe Hand"
    if hand_side:
        title += f" ({hand_side})"
    if win_name:
        try:
            cv2.putText(img, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(
                img,
                "rotate: J/L yaw  I/K pitch  U/O roll   zoom: +/-   reset: R   quit: Q/ESC",
                (10, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
            )
            cv2.imshow(win_name, img)
            # waitKey(1) for refresh; return False if user presses 'q'/'ESC'
            k = int(cv2.waitKey(1) & 0xFF)
            if k in (ord('q'), 27):
                return False, view
            # Key controls
            step_deg = 5.0
            if k == ord('j'):
                view.yaw_deg -= step_deg
            elif k == ord('l'):
                view.yaw_deg += step_deg
            elif k == ord('i'):
                view.pitch_deg -= step_deg
            elif k == ord('k'):
                view.pitch_deg += step_deg
            elif k == ord('u'):
                view.roll_deg -= step_deg
            elif k == ord('o'):
                view.roll_deg += step_deg
            elif k in (ord('+'), ord('=')):
                view.scale_px_per_m *= 1.1
            elif k in (ord('-'), ord('_')):
                view.scale_px_per_m /= 1.1
            elif k == ord('r'):
                view.yaw_deg = 0.0
                view.pitch_deg = 0.0
                view.roll_deg = 0.0
                view.scale_px_per_m = float(scale_px_per_m)
        except Exception:
            # likely no DISPLAY / headless
            return False, view
    return True, view


class _Hand21DViz3D:
    """
    真 3D 可视化：Open3D 点云 + 线段骨架。
    - 手腕(0)和五个指尖(4/8/12/16/20)标红
    - 其他点标绿
    - 鼠标可旋转/缩放/平移（Open3D 默认交互）
    """

    def __init__(self, win_name: str = "hand_21d_3d", axis_len_m: float = 0.10):
        self.win_name = str(win_name)
        self.axis_len_m = float(axis_len_m)
        self._vis = None  # type: Optional[Any]
        self._pcd = None  # type: Optional[Any]
        self._lines = None  # type: Optional[Any]
        self._inited = False

    def init(self) -> bool:
        if not _O3D_AVAILABLE or o3d is None:
            if _O3D_IMPORT_ERROR:
                print(f"⚠️  Open3D 导入失败: {_O3D_IMPORT_ERROR}")
            return False
        try:
            # Detect display availability early (common failure on headless / SSH without X forwarding)
            has_display = bool(os.environ.get("DISPLAY")) or bool(os.environ.get("WAYLAND_DISPLAY"))
            if not has_display:
                print("⚠️  Open3D 3D 可视化：未检测到 DISPLAY/WAYLAND_DISPLAY，可能是无桌面或 SSH 未开启 X 转发。")
                print("   - 解决：在有桌面的机器上跑；或 SSH 用 `-X/-Y`；或配置虚拟显示（xvfb）。")
                return False

            vis = o3d.visualization.Visualizer()
            ok = bool(vis.create_window(window_name=self.win_name, width=900, height=700, visible=True))
            if not ok:
                print("⚠️  Open3D 3D 可视化：create_window() 失败（通常是图形/GL 依赖缺失或显示环境异常）。")
                print("   - 你可以先自检：`echo $DISPLAY`、`glxinfo | head`（需要 mesa-utils）")
                print("   - 常见依赖：libgl1-mesa-glx/libgl1、libx11-6、libxi6、libxrandr2、libxinerama1、libxcursor1")
                return False
            self._vis = vis

            self._pcd = o3d.geometry.PointCloud()
            self._lines = o3d.geometry.LineSet()

            # Add geometry once
            self._vis.add_geometry(self._pcd)
            self._vis.add_geometry(self._lines)

            # Optional coordinate frame
            if self.axis_len_m > 0:
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.axis_len_m, origin=[0, 0, 0])
                self._vis.add_geometry(axis)

            self._inited = True
            return True
        except Exception:
            print("⚠️  Open3D 3D 可视化：初始化异常（可能是 GL/GUI 依赖问题）。")
            return False

    def close(self):
        try:
            if self._vis is not None:
                self._vis.destroy_window()
        except Exception:
            pass
        self._vis = None
        self._pcd = None
        self._lines = None
        self._inited = False

    def update(self, pts21: np.ndarray) -> bool:
        if not self._inited:
            if not self.init():
                return False
        if self._vis is None or self._pcd is None or self._lines is None or o3d is None:
            return False

        pts21 = np.asarray(pts21, dtype=np.float32).reshape(21, 3)

        try:
            # Points
            self._pcd.points = o3d.utility.Vector3dVector(pts21.astype(np.float64))
            tip_idxs = {0, 4, 8, 12, 16, 20}
            colors = np.zeros((21, 3), dtype=np.float64)
            for i in range(21):
                if i in tip_idxs:
                    colors[i] = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # red
                else:
                    colors[i] = np.array([0.0, 1.0, 0.0], dtype=np.float64)  # green
            self._pcd.colors = o3d.utility.Vector3dVector(colors)

            # Lines
            lines = np.asarray(_MP_HAND_CONNECTIONS, dtype=np.int32)
            self._lines.points = o3d.utility.Vector3dVector(pts21.astype(np.float64))
            self._lines.lines = o3d.utility.Vector2iVector(lines)
            self._lines.colors = o3d.utility.Vector3dVector(
                np.tile(np.array([[0.7, 0.7, 0.7]], dtype=np.float64), (lines.shape[0], 1))
            )

            self._vis.update_geometry(self._pcd)
            self._vis.update_geometry(self._lines)

            ok = bool(self._vis.poll_events())
            self._vis.update_renderer()
            if not ok:
                # window closed
                self.close()
                return False
            return True
        except Exception:
            # any rendering issue -> disable
            self.close()
            return False

def now_ms() -> int:
    """当前时间戳（毫秒）"""
    return int(time.time() * 1000)


try:
    import wujihandpy
except ImportError:
    print("❌ 错误: 未安装 wujihandpy，请先安装:")
    print("   pip install wujihandpy")
    sys.exit(1)

# 添加 wuji_retargeting 到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WUJI_RETARGETING_PATH = PROJECT_ROOT / "wuji_retargeting"
if str(WUJI_RETARGETING_PATH) not in sys.path:
    sys.path.insert(0, str(WUJI_RETARGETING_PATH))

try:
    from wuji_retargeting import WujiHandRetargeter
    from wuji_retargeting.mediapipe import apply_mediapipe_transformations
except ImportError as e:
    print(f"❌ 错误: 无法导入 wuji_retargeting: {e}")
    print("   请确保 wuji_retargeting 已正确安装")
    sys.exit(1)


# 26维手部关节名称（与 xrobot_utils.py 中的定义一致）
HAND_JOINT_NAMES_26 = [
    "Wrist", "Palm",
    "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
    "IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
    "MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip", 
    "RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
    "LittleMetacarpal", "LittleProximal", "LittleIntermediate", "LittleDistal", "LittleTip"
]

# 26维到21维 MediaPipe 格式的映射索引
# MediaPipe 格式: [Wrist, Thumb(4), Index(4), Middle(4), Ring(4), Pinky(4)]
# 26维格式: [Wrist, Palm, Thumb(4), Index(5), Middle(5), Ring(5), Pinky(5)]
MEDIAPIPE_MAPPING_26_TO_21 = [
    1,   # 0: Wrist -> Wrist
    2,   # 1: ThumbMetacarpal -> Thumb CMC
    3,   # 2: ThumbProximal -> Thumb MCP
    4,   # 3: ThumbDistal -> Thumb IP
    5,   # 4: ThumbTip -> Thumb Tip
    6,   # 5: IndexMetacarpal -> Index MCP
    7,   # 6: IndexProximal -> Index PIP
    8,   # 7: IndexIntermediate -> Index DIP
    10,  # 8: IndexTip -> Index Tip (跳过 IndexDistal)
    11,  # 9: MiddleMetacarpal -> Middle MCP
    12,  # 10: MiddleProximal -> Middle PIP
    13,  # 11: MiddleIntermediate -> Middle DIP
    15,  # 12: MiddleTip -> Middle Tip (跳过 MiddleDistal)
    16,  # 13: RingMetacarpal -> Ring MCP
    17,  # 14: RingProximal -> Ring PIP
    18,  # 15: RingIntermediate -> Ring DIP
    20,  # 16: RingTip -> Ring Tip (跳过 RingDistal)
    21,  # 17: LittleMetacarpal -> Pinky MCP
    22,  # 18: LittleProximal -> Pinky PIP
    23,  # 19: LittleIntermediate -> Pinky DIP
    25,  # 20: LittleTip -> Pinky Tip (跳过 LittleDistal)
]


def hand_26d_to_mediapipe_21d(hand_data_dict, hand_side="left", print_distances=False):
    """
    将26维手部追踪数据转换为21维 MediaPipe 格式
    
    Args:
        hand_data_dict: 字典，包含26个关节的数据
                      格式: {"LeftHandWrist": [[x,y,z], [qw,qx,qy,qz]], ...}
        hand_side: "left" 或 "right"
        print_distances: 是否打印手腕到指尖的距离
    
    Returns:
        numpy array of shape (21, 3) - MediaPipe 格式的手部关键点
    """
    hand_side_prefix = "LeftHand" if hand_side.lower() == "left" else "RightHand"
    
    # 提取26个关节的位置
    joint_positions_26 = np.zeros((26, 3), dtype=np.float32)
    
    for i, joint_name in enumerate(HAND_JOINT_NAMES_26):
        key = hand_side_prefix + joint_name
        if key in hand_data_dict:
            pos = hand_data_dict[key][0]  # [x, y, z]
            joint_positions_26[i] = pos
        else:
            # 如果缺少数据，使用零值
            joint_positions_26[i] = [0.0, 0.0, 0.0]
    
    # 使用映射索引转换为21维
    mediapipe_21d = joint_positions_26[MEDIAPIPE_MAPPING_26_TO_21]
    
    # 将腕部坐标设为0（作为原点）
    wrist_pos = mediapipe_21d[0].copy()  # 保存原始腕部位置
    mediapipe_21d = mediapipe_21d - wrist_pos  # 所有点相对于腕部
    
    # 其他坐标（除了腕部）乘以1.8倍
    scale_factor = 1.0
    mediapipe_21d[1:] = mediapipe_21d[1:] * scale_factor  # 索引1-20都乘以1.8
    # 腕部保持为0（索引0）

    # 计算并打印手腕到各指尖的距离（仅在需要时打印，避免实时环刷屏）
    if print_distances:
        print("大拇指位置: ", mediapipe_21d[4])
        print("食指位置: ", mediapipe_21d[8])
        print("中指位置: ", mediapipe_21d[12])
        print("无名指位置: ", mediapipe_21d[16])
        print("小拇指位置: ", mediapipe_21d[20])

        # MediaPipe 格式的指尖索引
        fingertip_indices = {
            "Thumb": 4,    # ThumbTip
            "Index": 8,    # IndexTip
            "Middle": 12,  # MiddleTip
            "Ring": 16,    # RingTip
            "Pinky": 20,   # PinkyTip
        }
        
        print("\n📏 手腕到各指尖的距离 (单位: 米):")
        print("-" * 50)
        wrist_pos_scaled = mediapipe_21d[0]  # 应该是 [0, 0, 0]
        for finger_name, tip_idx in fingertip_indices.items():
            tip_pos = mediapipe_21d[tip_idx]
            distance = np.linalg.norm(tip_pos - wrist_pos_scaled)
            print(f"  {finger_name:6s} (索引 {tip_idx:2d}): {distance*100:6.2f} cm ({distance:.4f} m)")
        print("-" * 50)

        # print(mediapipe_21d)
        # print("-" * 50)
        # print("-" * 50)
        # print(joint_positions_26)
        # print("-" * 50)
    
    return mediapipe_21d


def smooth_move(hand, controller, target_qpos, duration=0.1, steps=10):
    """
    平滑移动到某个 5×4 的关节目标
    
    Args:
        hand: wujihandpy.Hand 对象
        controller: wujihandpy 控制器对象
        target_qpos: numpy array of shape (5, 4)
        duration: 平滑移动持续时间（秒）
        steps: 平滑移动步数
    """
    target_qpos = target_qpos.reshape(5, 4)
    try:
        # cur = controller.get_joint_actual_position()
        cur = controller.read_joint_actual_position()
    except:
        cur = np.zeros((5, 4), dtype=np.float32)
    
    for t in np.linspace(0, 1, steps):
        q = cur * (1 - t) + target_qpos * t
        controller.set_joint_target_position(q)
        time.sleep(duration / steps)


class WujiHandRedisController:
    """从 Redis 读取手部追踪数据并控制 Wuji 手的控制器"""
    
    def __init__(
        self,
        redis_ip="localhost",
        hand_side="left",
        target_fps=50,
        smooth_enabled=True,
        smooth_steps=5,
        serial_number: str = "",
        viz_hand21d: bool = False,
        viz_hand21d_size: int = 640,
        viz_hand21d_scale: float = 1200.0,
        viz_hand21d_show_index: bool = False,
        viz_hand21d_3d: bool = False,
        viz_hand21d_3d_axis_len_m: float = 0.10,
    ):
        """
        Args:
            redis_ip: Redis 服务器 IP
            hand_side: "left" 或 "right"
            target_fps: 目标控制频率 (Hz)
            smooth_enabled: 是否启用平滑移动
            smooth_steps: 平滑移动步数
        """
        self.hand_side = hand_side.lower()
        assert self.hand_side in ["left", "right"], "hand_side must be 'left' or 'right'"
        
        self.target_fps = target_fps
        self.control_dt = 1.0 / target_fps
        self.smooth_enabled = smooth_enabled
        self.smooth_steps = smooth_steps
        self.serial_number = (serial_number or "").strip()

        # 21D 可视化参数（OpenCV）
        self.viz_hand21d = bool(viz_hand21d)
        self.viz_hand21d_size = int(viz_hand21d_size)
        self.viz_hand21d_scale = float(viz_hand21d_scale)
        self.viz_hand21d_show_index = bool(viz_hand21d_show_index)
        self._viz_ok = None  # None=unknown, True/False known after first draw
        self._viz_view = _HandVizView(scale_px_per_m=self.viz_hand21d_scale)

        # 真 3D 可视化（Open3D）
        self.viz_hand21d_3d = bool(viz_hand21d_3d)
        self.viz_hand21d_3d_axis_len_m = float(viz_hand21d_3d_axis_len_m)
        self._viz3d_ok = None
        self._viz3d = _Hand21DViz3D(win_name=f"hand_21d_3d_{self.hand_side}", axis_len_m=self.viz_hand21d_3d_axis_len_m)
        
        # 连接 Redis
        print(f"🔗 连接 Redis: {redis_ip}")
        try:
            self.redis_client = redis.Redis(host=redis_ip, port=6379, decode_responses=False)
            self.redis_client.ping()
            print("✅ Redis 连接成功")
        except Exception as e:
            print(f"❌ Redis 连接失败: {e}")
            raise
        
        # Redis 键名
        # - hand_tracking_*：来自 xrobot_teleop_to_robot_w_hand.py 的 26D dict（上游 action 输入）
        # - action_wuji_qpos_target_*：本脚本 retarget 后得到的 Wuji 关节目标（中间 action，便于复现/排障）
        # - state_wuji_hand_*：从硬件读取的实际关节位置（state 反馈）
        self.robot_key = "unitree_g1_with_hands"
        self.redis_key_hand_tracking = f"hand_tracking_{self.hand_side}_{self.robot_key}"
        self.redis_key_action_wuji_qpos_target = f"action_wuji_qpos_target_{self.hand_side}_{self.robot_key}"
        self.redis_key_state_wuji_hand = f"state_wuji_hand_{self.hand_side}_{self.robot_key}"
        self.redis_key_t_action_wuji_hand = f"t_action_wuji_hand_{self.hand_side}_{self.robot_key}"
        self.redis_key_t_state_wuji_hand = f"t_state_wuji_hand_{self.hand_side}_{self.robot_key}"
        # teleop 写入的 Wuji 模式开关：follow / hold / default
        self.redis_key_wuji_mode = f"wuji_hand_mode_{self.hand_side}_{self.robot_key}"
        
        # 初始化 Wuji 手
        print(f"🤖 初始化 Wuji {self.hand_side} 手...")
        if self.serial_number:
            print(f"🔌 使用 serial_number 选择设备: {self.serial_number}")
            self.hand = wujihandpy.Hand(serial_number=self.serial_number)
        else:
            self.hand = wujihandpy.Hand()
        self.hand.write_joint_enabled(True)
        self.controller = self.hand.realtime_controller(
            enable_upstream=True,
            filter=wujihandpy.filter.LowPass(cutoff_freq=10.0)
        )
        time.sleep(0.4)
        
        # 设置“零位”为全 0（用于回零/初始化 last_qpos）
        # 仍然读取一次实际关节位置以获得正确 shape，避免不同设备/固件返回维度差异
        # actual_pose = self.hand.get_joint_actual_position()
        actual_pose = self.hand.read_joint_actual_position()
        self.zero_pose = np.zeros_like(actual_pose)
        print(f"✅ Wuji {self.hand_side} 手初始化完成")
        
        # 初始化重定向器
        print(f"🔄 初始化 WujiHandRetargeter ({self.hand_side})...")
        self.retargeter = WujiHandRetargeter(hand_side=self.hand_side)
        print("✅ 重定向器初始化完成")
        
        # 状态变量
        self.last_qpos = self.zero_pose.copy()
        self.running = True
        self._cleaned_up = False
        self._stop_requested_by_signal = None
        self._frame_count = 0
        self._distance_printed = False
        self._has_received_data = False  # 标记是否收到过数据
        
        # 帧率监控（只统计有数据时的帧率）
        self._fps_start_time = None
        self._fps_data_frame_count = 0  # 只统计有数据时的帧数
        self._fps_print_interval = 100  # 每100帧打印一次帧率
        
    def get_hand_tracking_data_from_redis(self):
        """
        从 Redis 读取手部追踪数据（26维字典格式）
        检查数据是否新鲜（通过时间戳判断）
        
        Returns:
            tuple: (is_active, hand_data_dict) 或 (None, None)
        """
        try:
            # 尝试从 Redis 读取手部追踪数据
            data = self.redis_client.get(self.redis_key_hand_tracking)
            
            if data is None:
                # 调试：打印 Redis key
                if not hasattr(self, '_debug_key_printed'):
                    print(f"⚠️  Redis key '{self.redis_key_hand_tracking}' 不存在或为空")
                    self._debug_key_printed = True
                return None, None
            
            # 解析 JSON（注意：如果 decode_responses=False，data 是 bytes，需要 decode）
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            
            hand_data = json.loads(data)
            
            # 检查数据格式
            if isinstance(hand_data, dict):
                # 检查数据是否新鲜（通过时间戳）
                # 如果数据超过 0.5 秒没有更新，认为 teleop 已停止
                data_timestamp = hand_data.get("timestamp", 0)
                current_time_ms = int(time.time() * 1000)
                time_diff_ms = current_time_ms - data_timestamp
                
                # 如果时间差超过 500ms，认为数据过期
                if time_diff_ms > 500:
                    if not hasattr(self, '_debug_stale_printed'):
                        print(f"⚠️  数据过期 (时间差: {time_diff_ms}ms > 500ms)")
                        self._debug_stale_printed = True
                    return None, None
                
                # 检查 is_active 标志
                is_active = hand_data.get("is_active", False)
                if not is_active:
                    if not hasattr(self, '_debug_inactive_printed'):
                        print(f"⚠️  手部追踪数据 is_active=False")
                        self._debug_inactive_printed = True
                    return None, None
                
                # 提取手部数据（排除元数据）
                hand_dict = {k: v for k, v in hand_data.items() 
                           if k not in ["is_active", "timestamp"]}
                
                # 调试：第一次成功读取时打印
                if not hasattr(self, '_debug_success_printed'):
                    print(f"✅ 成功从 Redis 读取手部追踪数据 (key: {self.redis_key_hand_tracking}, "
                          f"关节数: {len(hand_dict)})")
                    self._debug_success_printed = True
                
                return is_active, hand_dict
            else:
                print(f"⚠️  数据格式错误: 期望 dict，得到 {type(hand_data)}")
                return None, None
                
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON 解析错误: {e}")
            return None, None
        except Exception as e:
            print(f"⚠️  读取 Redis 数据错误: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def run(self):
        """主控制循环"""
        print(f"\n🚀 开始控制循环 (目标频率: {self.target_fps} Hz)")
        print("按 Ctrl+C 退出\n")
        
        # 初始化帧率监控（只统计有数据时的帧率）
        self._fps_start_time = None
        self._fps_data_frame_count = 0
        
        def _handle_signal(signum, _frame):
            # 让控制环自然退出，进入 finally 的 cleanup，释放 USB 资源
            self._stop_requested_by_signal = signum
            self.running = False

        # 注意：当该进程作为“后台任务”运行时，Ctrl+C 往往不会送到它；
        # 但 shell 的 trap 通常会对它发 SIGTERM，所以我们需要处理 SIGTERM 来做安全退出。
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        # 1230 debug -- Yihe
        from rich.live import Live
        from rich.text import Text
        live = Live(refresh_per_second=10, auto_refresh=True)
        live.start()

        try:
            while self.running:
                loop_start = time.time()

                # 0) 读取模式（默认为 follow）
                try:
                    mode_raw = self.redis_client.get(self.redis_key_wuji_mode)
                    if isinstance(mode_raw, bytes):
                        mode_raw = mode_raw.decode("utf-8")
                    mode = str(mode_raw) if mode_raw is not None else "follow"
                except Exception:
                    mode = "follow"
                mode = mode.strip().lower()

                # # 1230 debug -- Yihe
                # # actual_qpos = self.hand.get_joint_actual_position()
                # # print(f"actual_qpos: {actual_qpos}")
                # actual_qpos = self.hand.read_joint_actual_position()
                # # print(f"actual_qpos: {actual_qpos}")
                # with np.printoptions(precision=4, suppress=True):
                #     live.update(Text(f"actual_qpos: {actual_qpos}"))

                # 0.1) default: 回零位；hold: 保持 last_qpos（两者都不依赖 tracking）
                if mode in ["default", "hold"]:
                    try:
                        target = self.zero_pose if mode == "default" else self.last_qpos
                        if target is None:
                            target = self.zero_pose

                        # 写 action target（便于录制/排障）
                        try:
                            self.redis_client.set(self.redis_key_action_wuji_qpos_target, json.dumps(target.reshape(-1).tolist()))
                            self.redis_client.set(self.redis_key_t_action_wuji_hand, now_ms())
                        except Exception:
                            pass

                        # 下发控制
                        if self.hand is not None and self.controller is not None:
                            if self.smooth_enabled:
                                smooth_move(self.hand, self.controller, target, duration=self.control_dt, steps=self.smooth_steps)
                            else:
                                self.controller.set_joint_target_position(target)

                            # 写 state（硬件实际位置）
                            try:
                                # actual_qpos = self.hand.get_joint_actual_position()
                                actual_qpos = self.hand.read_joint_actual_position()
                                self.redis_client.set(self.redis_key_state_wuji_hand, json.dumps(actual_qpos.reshape(-1).tolist()))
                                self.redis_client.set(self.redis_key_t_state_wuji_hand, now_ms())
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"⚠️  模式 {mode} 控制失败: {e}")

                    # 控制频率
                    elapsed = time.time() - loop_start
                    sleep_time = max(0, self.control_dt - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    continue
                
                # 从 Redis 读取手部追踪数据
                is_active, hand_data_dict = self.get_hand_tracking_data_from_redis()
                # print(f"is_active: {is_active}, hand_data_dict: {hand_data_dict}")
                
                if is_active and hand_data_dict is not None:
                    try:
                        # 初始化帧率统计（第一次收到数据时）
                        if self._fps_start_time is None:
                            self._fps_start_time = time.time()
                            self._fps_data_frame_count = 0
                        
                        # 1. 将26维转换为21维 MediaPipe 格式
                        # 只在第一次打印距离信息，然后退出
                        print_distances = False
                        mediapipe_21d = hand_26d_to_mediapipe_21d(hand_data_dict, self.hand_side, 
                                                                  print_distances=print_distances)
                        # 1.4 真 3D 可视化（Open3D，可选）
                        if self.viz_hand21d_3d:
                            ok3d = self._viz3d.update(mediapipe_21d)
                            if self._viz3d_ok is None:
                                self._viz3d_ok = bool(ok3d)
                                if not self._viz3d_ok:
                                    print("⚠️  21D 3D 可视化初始化失败（可能未安装 open3d 或无 GUI）。已自动关闭 3D 可视化。")
                                    self.viz_hand21d_3d = False
                            elif not ok3d:
                                print("🛑  21D 3D 可视化已退出（窗口关闭或异常）。继续控制环。")
                                self.viz_hand21d_3d = False
                        # 1.5 画 21D 手（可选）
                        if self.viz_hand21d:
                            ok, self._viz_view = _viz_draw_mediapipe_hand_21d(
                                mediapipe_21d,
                                win_name=f"hand_21d_{self.hand_side}",
                                canvas_size=self.viz_hand21d_size,
                                scale_px_per_m=self.viz_hand21d_scale,
                                flip_y=True,
                                show_index=self.viz_hand21d_show_index,
                                hand_side=self.hand_side,
                                view=self._viz_view,
                            )
                            # 第一次失败时给提示，并自动关闭可视化，避免刷异常
                            if self._viz_ok is None:
                                self._viz_ok = bool(ok)
                                if not self._viz_ok:
                                    print("⚠️  21D 可视化初始化失败（可能没有 GUI/未安装 opencv-python）。已自动关闭可视化。")
                                    self.viz_hand21d = False
                            elif not ok:
                                # 用户按 q/ESC 或窗口异常
                                print("🛑  21D 可视化已退出（q/ESC 或窗口异常）。继续控制环。")
                                self.viz_hand21d = False
                        if print_distances:
                            self._distance_printed = True
                            print("\n✅ 距离信息已打印，程序退出")
                            break  # 退出循环
                        
                        # 2. 应用 MediaPipe 变换
                        mediapipe_transformed = apply_mediapipe_transformations(
                            mediapipe_21d, 
                            hand_type=self.hand_side
                        )
                        
                        # 3. 使用 WujiHandRetargeter 进行重定向
                        retarget_result = self.retargeter.retarget(mediapipe_transformed)
                        wuji_20d = retarget_result.robot_qpos.reshape(5, 4)

                        # 3.5 写回 Redis：记录 Wuji 手 action/target（retarget 输出）
                        try:
                            self.redis_client.set(self.redis_key_action_wuji_qpos_target, json.dumps(wuji_20d.reshape(-1).tolist()))
                            self.redis_client.set(self.redis_key_t_action_wuji_hand, now_ms())
                        except Exception:
                            # 记录失败不影响控制环
                            pass
                        
                        # 4. 控制 Wuji 手（只在有新数据时发送命令）
                        if self.hand is not None and self.controller is not None:
                            if self.smooth_enabled:
                                smooth_move(self.hand, self.controller, wuji_20d, 
                                          duration=self.control_dt, steps=self.smooth_steps)
                            else:
                                self.controller.set_joint_target_position(wuji_20d)

                            # 4.5 写回 Redis：记录 Wuji 手 state（硬件实际位置）
                            # 注意：额外的 USB 读取可能带来开销/不稳定性；这里仅在 active 控制时读取一次
                            try:
                                # actual_qpos = self.hand.get_joint_actual_position()
                                actual_qpos = self.hand.read_joint_actual_position()
                                self.redis_client.set(self.redis_key_state_wuji_hand, json.dumps(actual_qpos.reshape(-1).tolist()))
                                self.redis_client.set(self.redis_key_t_state_wuji_hand, now_ms())
                            except Exception:
                                pass
                        
                        self.last_qpos = wuji_20d.copy()
                        self._has_received_data = True  # 标记已收到数据
                        self._frame_count += 1
                        
                        # 帧率统计（只统计有数据时的帧）
                        self._fps_data_frame_count += 1
                        if self._fps_data_frame_count >= self._fps_print_interval:
                            elapsed_time = time.time() - self._fps_start_time
                            actual_fps = self._fps_data_frame_count / elapsed_time
                            print(f"📊 实际数据帧率: {actual_fps:.2f} Hz (目标: {self.target_fps} Hz, "
                                  f"已处理 {self._fps_data_frame_count} 帧数据)")
                            # 重置计数器
                            self._fps_start_time = time.time()
                            self._fps_data_frame_count = 0
                        
                    except Exception as e:
                        print(f"⚠️  处理手部数据失败: {e}")
                        import traceback
                        traceback.print_exc()
                        # 不发送命令，等待下次数据
                else:
                    # 如果没有数据，不发送任何命令（避免重复发送 last_qpos）
                    # 只在第一次没有数据时打印提示
                    if not self._has_received_data and self._frame_count == 0:
                        print("⏳ 等待手部追踪数据...")
                    self._frame_count += 1
                
                # 控制频率
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.control_dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        print("\n🛑 正在关闭控制器并失能电机...")
        try:
            # 平滑回到零位
            # 如果是 SIGTERM 触发退出，尽量缩短回零时间，避免拖太久导致再次被强杀
            if self._stop_requested_by_signal == signal.SIGTERM:
                smooth_move(self.hand, self.controller, self.zero_pose, duration=0.2, steps=10)
            else:
                smooth_move(self.hand, self.controller, self.zero_pose, duration=1.0, steps=50)
            print("✅ 已回到零位")
        except:
            pass
        
        try:
            self.controller.close()
            self.hand.write_joint_enabled(False)
            print("✅ 控制器已关闭")
        except:
            pass

        # close viz windows (optional)
        try:
            if self._viz3d is not None:
                self._viz3d.close()
        except Exception:
            pass
        try:
            if _CV2_AVAILABLE and cv2 is not None:
                cv2.destroyAllWindows()
        except Exception:
            pass
        
        print("✅ 退出完成")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="从 Redis 读取手部追踪数据并控制 Wuji 灵巧手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 控制左手
  python server_wuji_hand_redis.py --hand_side left --redis_ip localhost

  # 控制右手，50Hz 频率
  python server_wuji_hand_redis.py --hand_side right --target_fps 50

  # 禁用平滑移动
  python server_wuji_hand_redis.py --hand_side left --no_smooth
        """
    )
    
    parser.add_argument(
        "--hand_side",
        type=str,
        default="left",
        choices=["left", "right"],
        help="控制左手或右手 (默认: left)"
    )
    
    parser.add_argument(
        "--redis_ip",
        type=str,
        default="localhost",
        help="Redis 服务器 IP (默认: localhost)"
    )
    
    parser.add_argument(
        "--target_fps",
        type=int,
        default=50,
        help="目标控制频率 (Hz) (默认: 50)"
    )
    
    parser.add_argument(
        "--no_smooth",
        action="store_true",
        help="禁用平滑移动（直接设置目标位置）"
    )
    
    parser.add_argument(
        "--smooth_steps",
        type=int,
        default=5,
        help="平滑移动步数 (默认: 5)"
    )

    parser.add_argument(
        "--serial_number",
        type=str,
        default="",
        help="可选：指定 Wuji 手设备序列号，用于多设备环境下筛选正确设备（例如 337238793233）",
    )

    # 21D 手关键点可视化（OpenCV）
    parser.add_argument(
        "--viz_hand21d",
        action="store_true",
        help="实时画 21D MediaPipe 手关键点（带连线；手腕+指尖标红）。需要 opencv-python 且有桌面环境。",
    )
    parser.add_argument(
        "--viz_hand21d_size",
        type=int,
        default=640,
        help="可视化窗口尺寸（像素，正方形）(默认: 640)",
    )
    parser.add_argument(
        "--viz_hand21d_scale",
        type=float,
        default=1200.0,
        help="投影缩放（像素/米，默认: 1200）",
    )
    parser.add_argument(
        "--viz_hand21d_show_index",
        action="store_true",
        help="在关键点旁显示索引号（0-20）",
    )

    # 21D 真 3D 可视化（Open3D）
    parser.add_argument(
        "--viz_hand21d_3d",
        action="store_true",
        help="真 3D 画 21D MediaPipe 手关键点（可鼠标旋转/缩放/平移）。需要 open3d 且有桌面环境。",
    )
    parser.add_argument(
        "--viz_hand21d_3d_axis_len_m",
        type=float,
        default=0.10,
        help="3D 坐标轴长度（米，默认 0.10；设为 0 可关闭坐标轴）",
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    print("=" * 60)
    print("Wuji Hand Controller via Redis (26D -> 21D MediaPipe -> Retarget)")
    print("=" * 60)
    print(f"手部: {args.hand_side}")
    print(f"Redis IP: {args.redis_ip}")
    print(f"目标频率: {args.target_fps} Hz")
    print(f"平滑移动: {'禁用' if args.no_smooth else '启用'}")
    if not args.no_smooth:
        print(f"平滑步数: {args.smooth_steps}")
    print("=" * 60)
    
    try:
        controller = WujiHandRedisController(
            redis_ip=args.redis_ip,
            hand_side=args.hand_side,
            target_fps=args.target_fps,
            smooth_enabled=not args.no_smooth,
            smooth_steps=args.smooth_steps,
            serial_number=args.serial_number,
            viz_hand21d=args.viz_hand21d,
            viz_hand21d_size=args.viz_hand21d_size,
            viz_hand21d_scale=args.viz_hand21d_scale,
            viz_hand21d_show_index=args.viz_hand21d_show_index,
            viz_hand21d_3d=args.viz_hand21d_3d,
            viz_hand21d_3d_axis_len_m=args.viz_hand21d_3d_axis_len_m,
        )
        controller.run()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()