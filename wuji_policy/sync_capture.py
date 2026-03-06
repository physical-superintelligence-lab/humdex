#!/usr/bin/env python3
"""
同步采集视频流和动捕数据
- 视频: MP4 格式
- 动捕: 原始 3D Pose (位置 + 四元数)
- 每 2 分钟自动分块保存
按 Ctrl+C 停止录制
"""
import csv
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import geort

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# 添加 SDK 路径
sys.path.insert(0, str(Path(__file__).parent / "DataRead_Python_Linux_SDK/DataRead_Python_Demo"))
from vdmocapsdk_dataread import *
from vdmocapsdk_nodelist import *

# ==================== 配置参数 ====================
CONFIG = {
    # 动捕配置
    "mocap_index": 0,
    "mocap_dst_ip": "192.168.1.112",
    "mocap_dst_port": 7000,
    "mocap_local_port": 0,
    "mocap_world_space": 0,
    
    # 视频配置
    "video_device": "/dev/video1",
    "video_width": 3840,
    "video_height": 1080,
    "video_fps": 30,
    
    # 输出配置
    "output_dir": "./captured_data",
    "chunk_duration_sec": 120,  # 每 2 分钟分块
}

# 默认骨架 (用于 SDK 设置)
INITIAL_POSITION_BODY = [
    [0, 0, 1.022], [0.074, 0, 1.002], [0.097, 0, 0.593], [0.104, 0, 0.111],
    [0.114, 0.159, 0.005], [-0.074, 0, 1.002], [-0.097, 0.001, 0.593],
    [-0.104, 0, 0.111], [-0.114, 0.158, 0.004], [0, 0.033, 1.123],
    [0, 0.03, 1.246], [0, 0.014, 1.362], [0, -0.048, 1.475],
    [0, -0.048, 1.549], [0, -0.016, 1.682], [0.071, -0.061, 1.526],
    [0.178, -0.061, 1.526], [0.421, -0.061, 1.526], [0.682, -0.061, 1.526],
    [-0.071, -0.061, 1.526], [-0.178, -0.061, 1.526], [-0.421, -0.061, 1.526],
    [-0.682, -0.061, 1.526],
]

INITIAL_POSITION_HAND_RIGHT = [
    [0.682, -0.061, 1.526], [0.71, -0.024, 1.526], [0.728, -0.008, 1.526],
    [0.755, 0.013, 1.526], [0.707, -0.05, 1.526], [0.761, -0.024, 1.525],
    [0.812, -0.023, 1.525], [0.837, -0.022, 1.525], [0.709, -0.058, 1.526],
    [0.764, -0.046, 1.528], [0.816, -0.046, 1.528], [0.845, -0.046, 1.528],
    [0.709, -0.064, 1.526], [0.761, -0.069, 1.527], [0.812, -0.069, 1.527],
    [0.835, -0.069, 1.527], [0.708, -0.072, 1.526], [0.755, -0.089, 1.522],
    [0.791, -0.089, 1.522], [0.81, -0.089, 1.522],
]

INITIAL_POSITION_HAND_LEFT = [
    [-0.682, -0.061, 1.526], [-0.71, -0.024, 1.526], [-0.728, -0.008, 1.526],
    [-0.755, 0.013, 1.526], [-0.707, -0.05, 1.526], [-0.761, -0.024, 1.525],
    [-0.812, -0.023, 1.525], [-0.837, -0.022, 1.525], [-0.709, -0.058, 1.526],
    [-0.764, -0.046, 1.528], [-0.816, -0.046, 1.528], [-0.845, -0.046, 1.528],
    [-0.709, -0.064, 1.526], [-0.761, -0.069, 1.527], [-0.812, -0.069, 1.527],
    [-0.835, -0.069, 1.527], [-0.708, -0.072, 1.526], [-0.755, -0.089, 1.522],
    [-0.791, -0.089, 1.522], [-0.81, -0.089, 1.522],
]

# 全局停止标志
STOP_FLAG = False


class SyncCapture:
    def __init__(self, config):
        self.config = config
        self.running = False
        
        # 时间
        self.session_start_time = None
        self.chunk_start_time = None
        self.chunk_index = 0
        
        # 动捕数据
        self.mocap_file = None
        self.mocap_writer = None
        self.mocap_frame_count = 0
        self.mocap_frequency = 60
        self.total_mocap_frames = 0
        
        # 视频数据
        self.video_frame_count = 0
        self.video_ts_file = None
        self.total_video_frames = 0
        
        # 输出
        self.output_dir = None
        self.session_name = None
        
        # GStreamer
        Gst.init(None)
        self.pipeline = None
        
        # 线程同步
        self.lock = threading.Lock()
    
    def _get_chunk_paths(self):
        """获取当前 chunk 的文件路径"""
        prefix = f"{self.session_name}_chunk{self.chunk_index:03d}"
        return {
            "video": self.output_dir / f"{prefix}.mp4",
            "video_ts": self.output_dir / f"{prefix}_video_timestamps.csv",
            "mocap": self.output_dir / f"{prefix}_pose.csv",
        }
    
    def _create_mocap_header(self):
        """创建动捕 CSV 表头"""
        header = ["timestamp", "timestamp_ns", "frame_index", "frequency"]
        
        # 身体 23 个关节: 位置 xyz + 四元数 wxyz
        for name in NAMES_JOINT_BODY:
            header.extend([
                f"body_{name}_px", f"body_{name}_py", f"body_{name}_pz",
                f"body_{name}_qw", f"body_{name}_qx", f"body_{name}_qy", f"body_{name}_qz"
            ])
        
        # 右手 20 个关节
        for name in NAMES_JOINT_HAND_RIGHT:
            header.extend([
                f"rhand_{name}_px", f"rhand_{name}_py", f"rhand_{name}_pz",
                f"rhand_{name}_qw", f"rhand_{name}_qx", f"rhand_{name}_qy", f"rhand_{name}_qz"
            ])
        
        # 左手 20 个关节
        for name in NAMES_JOINT_HAND_LEFT:
            header.extend([
                f"lhand_{name}_px", f"lhand_{name}_py", f"lhand_{name}_pz",
                f"lhand_{name}_qw", f"lhand_{name}_qx", f"lhand_{name}_qy", f"lhand_{name}_qz"
            ])
        
        return header
    
    def _extract_mocap_row(self, mocap_data, timestamp, timestamp_ns):
        """提取一帧动捕数据"""
        row = [f"{timestamp:.6f}", str(timestamp_ns), mocap_data.frameIndex, mocap_data.frequency]
        
        # 身体数据
        for i in range(LENGTH_BODY):
            # 位置 (米)
            row.extend([
                f"{mocap_data.position_body[i][0]:.6f}",
                f"{mocap_data.position_body[i][1]:.6f}",
                f"{mocap_data.position_body[i][2]:.6f}",
            ])
            # 四元数 (wxyz)
            row.extend([
                f"{mocap_data.quaternion_body[i][0]:.6f}",
                f"{mocap_data.quaternion_body[i][1]:.6f}",
                f"{mocap_data.quaternion_body[i][2]:.6f}",
                f"{mocap_data.quaternion_body[i][3]:.6f}",
            ])
        
        # 右手数据
        for i in range(LENGTH_HAND):
            row.extend([
                f"{mocap_data.position_rHand[i][0]:.6f}",
                f"{mocap_data.position_rHand[i][1]:.6f}",
                f"{mocap_data.position_rHand[i][2]:.6f}",
            ])
            row.extend([
                f"{mocap_data.quaternion_rHand[i][0]:.6f}",
                f"{mocap_data.quaternion_rHand[i][1]:.6f}",
                f"{mocap_data.quaternion_rHand[i][2]:.6f}",
                f"{mocap_data.quaternion_rHand[i][3]:.6f}",
            ])
        
        # 左手数据
        for i in range(LENGTH_HAND):
            row.extend([
                f"{mocap_data.position_lHand[i][0]:.6f}",
                f"{mocap_data.position_lHand[i][1]:.6f}",
                f"{mocap_data.position_lHand[i][2]:.6f}",
            ])
            row.extend([
                f"{mocap_data.quaternion_lHand[i][0]:.6f}",
                f"{mocap_data.quaternion_lHand[i][1]:.6f}",
                f"{mocap_data.quaternion_lHand[i][2]:.6f}",
                f"{mocap_data.quaternion_lHand[i][3]:.6f}",
            ])
        
        return row
    
    def _start_new_chunk(self):
        """开始新的 chunk"""
        # 关闭当前文件
        if self.mocap_file:
            self.mocap_file.close()
        if self.video_ts_file:
            self.video_ts_file.close()
        if self.pipeline:
            self.pipeline.send_event(Gst.Event.new_eos())
            time.sleep(0.5)
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
        
        # 增加 chunk 索引
        if self.mocap_frame_count > 0 or self.video_frame_count > 0:
            self.chunk_index += 1
        
        # 重置计数
        self.mocap_frame_count = 0
        self.video_frame_count = 0
        self.chunk_start_time = time.time()
        
        # 创建新文件
        paths = self._get_chunk_paths()
        
        # 动捕文件
        self.mocap_file = open(paths["mocap"], 'w', newline='')
        self.mocap_writer = csv.writer(self.mocap_file)
        self.mocap_writer.writerow(self._create_mocap_header())
        
        # 视频时间戳文件
        self.video_ts_file = open(paths["video_ts"], 'w')
        self.video_ts_file.write("frame_index,pts_ns,system_time,system_time_ns,offset_from_chunk_start\n")
        
        # 视频管道
        self._setup_gstreamer(str(paths["video"]))
        self.pipeline.set_state(Gst.State.PLAYING)
        
        print(f"\n开始 chunk {self.chunk_index}...")
    
    def _setup_gstreamer(self, output_path):
        """设置 GStreamer 管道"""
        cfg = self.config
        pipeline_str = (
            f"v4l2src device={cfg['video_device']} ! "
            f"video/x-h264,width={cfg['video_width']},height={cfg['video_height']},"
            f"framerate={cfg['video_fps']}/1 ! "
            "tee name=t ! "
            "queue ! h264parse ! mp4mux ! "
            f"filesink location={output_path} "
            "t. ! queue ! appsink name=sink emit-signals=True sync=False drop=True max-buffers=1"
        )
        
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.sink = self.pipeline.get_by_name('sink')
        self.sink.connect("new-sample", self._on_video_sample)
    
    def _on_video_sample(self, sink):
        """视频帧回调"""
        global STOP_FLAG
        if STOP_FLAG or not self.running:
            return Gst.FlowReturn.EOS
        
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR
        
        buf = sample.get_buffer()
        pts_ns = buf.pts
        system_time = time.time()
        system_time_ns = time.time_ns()
        
        with self.lock:
            if self.video_ts_file and not self.video_ts_file.closed:
                offset = system_time - self.chunk_start_time if self.chunk_start_time else 0
                self.video_ts_file.write(
                    f"{self.video_frame_count},{pts_ns},{system_time:.6f},"
                    f"{system_time_ns},{offset:.6f}\n"
                )
                self.video_frame_count += 1
                self.total_video_frames += 1
        
        return Gst.FlowReturn.OK
    
    def _mocap_thread(self):
        """动捕采集线程"""
        global STOP_FLAG
        cfg = self.config
        mocap_data = MocapData()
        
        # 打开 UDP
        if not udp_is_open(cfg["mocap_index"]):
            if not udp_open(cfg["mocap_index"], cfg["mocap_local_port"]):
                print("错误: 无法打开动捕UDP端口")
                return
        
        # 设置骨架
        udp_set_position_in_initial_tpose(
            cfg["mocap_index"], cfg["mocap_dst_ip"], cfg["mocap_dst_port"],
            cfg["mocap_world_space"], INITIAL_POSITION_BODY,
            INITIAL_POSITION_HAND_RIGHT, INITIAL_POSITION_HAND_LEFT
        )
        
        # 连接
        print("正在连接动捕服务器...")
        if not udp_send_request_connect(cfg["mocap_index"], cfg["mocap_dst_ip"], cfg["mocap_dst_port"]):
            print("错误: 无法连接动捕服务器")
            return
        
        print("动捕连接成功!")
        
        last_frame_index = -1
        
        while self.running and not STOP_FLAG:
            if udp_recv_mocap_data(cfg["mocap_index"], cfg["mocap_dst_ip"], 
                                   cfg["mocap_dst_port"], mocap_data):
                if mocap_data.isUpdate and mocap_data.frameIndex != last_frame_index:
                    timestamp = time.time()
                    timestamp_ns = time.time_ns()
                    
                    with self.lock:
                        if self.mocap_writer:
                            row = self._extract_mocap_row(mocap_data, timestamp, timestamp_ns)
                            self.mocap_writer.writerow(row)
                            self.mocap_frame_count += 1
                            self.total_mocap_frames += 1
                            self.mocap_frequency = mocap_data.frequency if mocap_data.frequency > 0 else 60
                            
                            # 每 100 帧刷新一次
                            if self.mocap_frame_count % 100 == 0:
                                self.mocap_file.flush()
                    
                    last_frame_index = mocap_data.frameIndex
            else:
                time.sleep(0.0001)
        
        # 关闭连接
        try:
            time.sleep(0.1)
            udp_remove(cfg["mocap_index"], cfg["mocap_dst_ip"], cfg["mocap_dst_port"])
            time.sleep(0.1)
            udp_close(cfg["mocap_index"])
        except Exception as e:
            print(f"关闭动捕连接时出错: {e}")
    
    def _chunk_manager_thread(self):
        """分块管理线程"""
        global STOP_FLAG
        chunk_duration = self.config["chunk_duration_sec"]
        
        while self.running and not STOP_FLAG:
            time.sleep(1)
            elapsed = time.time() - self.chunk_start_time
            if elapsed >= chunk_duration:
                with self.lock:
                    self._start_new_chunk()
    
    def _display_thread(self):
        """显示状态线程"""
        global STOP_FLAG
        
        while self.running and not STOP_FLAG:
            elapsed = time.time() - self.session_start_time
            chunk_elapsed = time.time() - self.chunk_start_time
            
            with self.lock:
                v_total = self.total_video_frames
                m_total = self.total_mocap_frames
                v_chunk = self.video_frame_count
                m_chunk = self.mocap_frame_count
            
            v_fps = v_total / elapsed if elapsed > 0 else 0
            m_fps = m_total / elapsed if elapsed > 0 else 0
            chunk_remain = self.config["chunk_duration_sec"] - chunk_elapsed
            
            print(f"\rChunk {self.chunk_index}: 视频={v_chunk} 动捕={m_chunk} | "
                  f"总计: 视频={v_total}({v_fps:.1f}fps) 动捕={m_total}({m_fps:.1f}fps) | "
                  f"下次保存: {chunk_remain:.0f}s | 总时长: {elapsed:.1f}s", 
                  end="", flush=True)
            
            time.sleep(0.5)
    
    def start(self):
        """开始采集"""
        global STOP_FLAG
        STOP_FLAG = False
        
        print("=" * 70)
        print("同步采集器 - 视频 (MP4) + 动捕 (3D Pose)")
        print("=" * 70)
        print(f"视频: {self.config['video_device']} @ "
              f"{self.config['video_width']}x{self.config['video_height']}@{self.config['video_fps']}fps")
        print(f"动捕: {self.config['mocap_dst_ip']}:{self.config['mocap_dst_port']}")
        print(f"分块: 每 {self.config['chunk_duration_sec']} 秒自动保存")
        print("动捕数据: 原始 3D Pose (位置 xyz + 四元数 wxyz)")
        print("按 Ctrl+C 停止录制")
        print("=" * 70)
        
        # 设置输出目录
        self.output_dir = Path(self.config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.running = True
        self.session_start_time = time.time()
        self.chunk_start_time = time.time()
        
        # 创建第一个 chunk 的文件
        paths = self._get_chunk_paths()
        
        self.mocap_file = open(paths["mocap"], 'w', newline='')
        self.mocap_writer = csv.writer(self.mocap_file)
        self.mocap_writer.writerow(self._create_mocap_header())
        
        self.video_ts_file = open(paths["video_ts"], 'w')
        self.video_ts_file.write("frame_index,pts_ns,system_time,system_time_ns,offset_from_chunk_start\n")
        
        self._setup_gstreamer(str(paths["video"]))
        self.pipeline.set_state(Gst.State.PLAYING)
        
        # 启动线程
        threads = [
            threading.Thread(target=self._mocap_thread, daemon=True),
            threading.Thread(target=self._chunk_manager_thread, daemon=True),
            threading.Thread(target=self._display_thread, daemon=True),
        ]
        
        for t in threads:
            t.start()
        
        print(f"\n输出目录: {self.output_dir}")
        print("开始录制...\n")
        
        try:
            while not STOP_FLAG:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        
        self._cleanup()
    
    def _cleanup(self):
        """清理并保存"""
        global STOP_FLAG
        STOP_FLAG = True
        self.running = False
        
        print("\n\n正在停止并保存...")
        time.sleep(0.5)
        
        # 停止视频管道
        try:
            if self.pipeline:
                self.pipeline.send_event(Gst.Event.new_eos())
                bus = self.pipeline.get_bus()
                bus.timed_pop_filtered(2 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
                self.pipeline.set_state(Gst.State.NULL)
        except Exception as e:
            print(f"停止视频管道时出错: {e}")
        
        # 关闭文件
        try:
            if self.mocap_file:
                self.mocap_file.close()
            if self.video_ts_file:
                self.video_ts_file.close()
        except:
            pass
        
        print(f"\n{'='*70}")
        print(f"录制完成!")
        print(f"总 chunks: {self.chunk_index + 1}")
        print(f"总视频帧: {self.total_video_frames}")
        print(f"总动捕帧: {self.total_mocap_frames}")
        print(f"输出目录: {self.output_dir}")
        print(f"{'='*70}")


def signal_handler(signum, frame):
    global STOP_FLAG
    STOP_FLAG = True


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    capture = SyncCapture(CONFIG)
    capture.start()


if __name__ == "__main__":
    main()
