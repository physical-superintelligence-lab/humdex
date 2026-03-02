#!/usr/bin/env python3
"""
Sync capture: video (MP4) + 3D pose (body + hands).
Chunk duration configurable (e.g. 2 min). Press Ctrl+C to stop.
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

# VDMocap SDK
sys.path.insert(0, str(Path(__file__).parent / "DataRead_Python_Linux_SDK/DataRead_Python_Demo"))
from vdmocapsdk_dataread import *
from vdmocapsdk_nodelist import *

# ==================== Configuration ====================
CONFIG = {
    # Mocap
    "mocap_index": 0,
    "mocap_dst_ip": "192.168.1.112",
    "mocap_dst_port": 7000,
    "mocap_local_port": 0,
    "mocap_world_space": 0,

    # Video
    "video_device": "/dev/video1",
    "video_width": 3840,
    "video_height": 1080,
    "video_fps": 30,

    # Output
    "output_dir": "./captured_data",
    "chunk_duration_sec": 120,  # 2 min per chunk
}

# Fallback initial pose (SDK default)
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

# Global stop flag for signal handler
STOP_FLAG = False


class SyncCapture:
    def __init__(self, config):
        self.config = config
        self.running = False
        
        # Timing
        self.session_start_time = None
        self.chunk_start_time = None
        self.chunk_index = 0
        
        # Mocap
        self.mocap_file = None
        self.mocap_writer = None
        self.mocap_frame_count = 0
        self.mocap_frequency = 60
        self.total_mocap_frames = 0
        
        # Video
        self.video_frame_count = 0
        self.video_ts_file = None
        self.total_video_frames = 0
        
        # Output
        self.output_dir = None
        self.session_name = None
        
        # GStreamer
        Gst.init(None)
        self.pipeline = None
        
        # Thread safety
        self.lock = threading.Lock()
    
    def _get_chunk_paths(self):
        """Return file paths for current chunk."""
        prefix = f"{self.session_name}_chunk{self.chunk_index:03d}"
        return {
            "video": self.output_dir / f"{prefix}.mp4",
            "video_ts": self.output_dir / f"{prefix}_video_timestamps.csv",
            "mocap": self.output_dir / f"{prefix}_pose.csv",
        }
    
    def _create_mocap_header(self):
        """Build CSV header for mocap data."""
        header = ["timestamp", "timestamp_ns", "frame_index", "frequency"]
        
        # 23 body joints: xyz + quat wxyz
        for name in NAMES_JOINT_BODY:
            header.extend([
                f"body_{name}_px", f"body_{name}_py", f"body_{name}_pz",
                f"body_{name}_qw", f"body_{name}_qx", f"body_{name}_qy", f"body_{name}_qz"
            ])
        
        # 20 right-hand joints
        for name in NAMES_JOINT_HAND_RIGHT:
            header.extend([
                f"rhand_{name}_px", f"rhand_{name}_py", f"rhand_{name}_pz",
                f"rhand_{name}_qw", f"rhand_{name}_qx", f"rhand_{name}_qy", f"rhand_{name}_qz"
            ])
        
        # 20 left-hand joints
        for name in NAMES_JOINT_HAND_LEFT:
            header.extend([
                f"lhand_{name}_px", f"lhand_{name}_py", f"lhand_{name}_pz",
                f"lhand_{name}_qw", f"lhand_{name}_qx", f"lhand_{name}_qy", f"lhand_{name}_qz"
            ])
        
        return header
    
    def _extract_mocap_row(self, mocap_data, timestamp, timestamp_ns):
        """Extract one CSV row from mocap_data."""
        row = [f"{timestamp:.6f}", str(timestamp_ns), mocap_data.frameIndex, mocap_data.frequency]
        
        # Body position (xyz)
        for i in range(LENGTH_BODY):
            row.extend([
                f"{mocap_data.position_body[i][0]:.6f}",
                f"{mocap_data.position_body[i][1]:.6f}",
                f"{mocap_data.position_body[i][2]:.6f}",
            ])
            # Quaternion (wxyz)
            row.extend([
                f"{mocap_data.quaternion_body[i][0]:.6f}",
                f"{mocap_data.quaternion_body[i][1]:.6f}",
                f"{mocap_data.quaternion_body[i][2]:.6f}",
                f"{mocap_data.quaternion_body[i][3]:.6f}",
            ])
        
        # Right hand
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
        
        # Left hand
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
        """Start a new chunk (close previous, open new files)."""
        # Close previous chunk
        if self.mocap_file:
            self.mocap_file.close()
        if self.video_ts_file:
            self.video_ts_file.close()
        if self.pipeline:
            self.pipeline.send_event(Gst.Event.new_eos())
            time.sleep(0.5)
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
        
        # Advance chunk index
        if self.mocap_frame_count > 0 or self.video_frame_count > 0:
            self.chunk_index += 1
        
        # Reset counters
        self.mocap_frame_count = 0
        self.video_frame_count = 0
        self.chunk_start_time = time.time()
        
        # Open new files
        paths = self._get_chunk_paths()
        
        # Mocap CSV
        self.mocap_file = open(paths["mocap"], 'w', newline='')
        self.mocap_writer = csv.writer(self.mocap_file)
        self.mocap_writer.writerow(self._create_mocap_header())
        
        # Video timestamps
        self.video_ts_file = open(paths["video_ts"], 'w')
        self.video_ts_file.write("frame_index,pts_ns,system_time,system_time_ns,offset_from_chunk_start\n")
        
        # GStreamer pipeline
        self._setup_gstreamer(str(paths["video"]))
        self.pipeline.set_state(Gst.State.PLAYING)
        
        print(f"\nStarted chunk {self.chunk_index}...")
    
    def _setup_gstreamer(self, output_path):
        """Set up GStreamer pipeline for video capture."""
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
        """GStreamer appsink callback for each video frame."""
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
        """Thread: receive mocap data over UDP and write to CSV."""
        global STOP_FLAG
        cfg = self.config
        mocap_data = MocapData()
        
        # Open UDP socket
        if not udp_is_open(cfg["mocap_index"]):
            if not udp_open(cfg["mocap_index"], cfg["mocap_local_port"]):
                print("[ERROR] Failed to open UDP socket.")
                return
        
        # Set initial pose
        udp_set_position_in_initial_tpose(
            cfg["mocap_index"], cfg["mocap_dst_ip"], cfg["mocap_dst_port"],
            cfg["mocap_world_space"], INITIAL_POSITION_BODY,
            INITIAL_POSITION_HAND_RIGHT, INITIAL_POSITION_HAND_LEFT
        )
        
        # Request connect
        print("Connecting to mocap server...")
        if not udp_send_request_connect(cfg["mocap_index"], cfg["mocap_dst_ip"], cfg["mocap_dst_port"]):
            print("[ERROR] Failed to connect to mocap server.")
            return
        
        print("Connected!")
        
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
                            
                            # Flush every 100 frames
                            if self.mocap_frame_count % 100 == 0:
                                self.mocap_file.flush()
                    
                    last_frame_index = mocap_data.frameIndex
            else:
                time.sleep(0.0001)
        
        # Disconnect
        try:
            time.sleep(0.1)
            udp_remove(cfg["mocap_index"], cfg["mocap_dst_ip"], cfg["mocap_dst_port"])
            time.sleep(0.1)
            udp_close(cfg["mocap_index"])
        except Exception as e:
            print(f"[WARN] UDP cleanup error: {e}")
    
    def _chunk_manager_thread(self):
        """Thread: rotate to new chunk when duration elapsed."""
        global STOP_FLAG
        chunk_duration = self.config["chunk_duration_sec"]
        
        while self.running and not STOP_FLAG:
            time.sleep(1)
            elapsed = time.time() - self.chunk_start_time
            if elapsed >= chunk_duration:
                with self.lock:
                    self._start_new_chunk()
    
    def _display_thread(self):
        """Thread: print progress (video/mocap counts, FPS, elapsed)."""
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
            
            print(f"\rChunk {self.chunk_index}: video={v_chunk} mocap={m_chunk} | "
                  f"Total: video={v_total}({v_fps:.1f}fps) mocap={m_total}({m_fps:.1f}fps) | "
                  f"Chunk remaining: {chunk_remain:.0f}s | Elapsed: {elapsed:.1f}s", 
                  end="", flush=True)
            
            time.sleep(0.5)
    
    def start(self):
        """Start sync capture (video + mocap)."""
        global STOP_FLAG
        STOP_FLAG = False
        
        print("=" * 70)
        print("Sync capture: video (MP4) + 3D pose")
        print("=" * 70)
        print(f"Video: {self.config['video_device']} @ "
              f"{self.config['video_width']}x{self.config['video_height']}@{self.config['video_fps']}fps")
        print(f"Mocap: {self.config['mocap_dst_ip']}:{self.config['mocap_dst_port']}")
        print(f"Chunk duration: {self.config['chunk_duration_sec']} sec")
        print("Pose format: 3D (xyz + quat wxyz)")
        print("Press Ctrl+C to stop.")
        print("=" * 70)
        
        # Create output dir
        self.output_dir = Path(self.config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.running = True
        self.session_start_time = time.time()
        self.chunk_start_time = time.time()
        
        # Open first chunk
        paths = self._get_chunk_paths()
        
        self.mocap_file = open(paths["mocap"], 'w', newline='')
        self.mocap_writer = csv.writer(self.mocap_file)
        self.mocap_writer.writerow(self._create_mocap_header())
        
        self.video_ts_file = open(paths["video_ts"], 'w')
        self.video_ts_file.write("frame_index,pts_ns,system_time,system_time_ns,offset_from_chunk_start\n")
        
        self._setup_gstreamer(str(paths["video"]))
        self.pipeline.set_state(Gst.State.PLAYING)
        
        # Start threads
        threads = [
            threading.Thread(target=self._mocap_thread, daemon=True),
            threading.Thread(target=self._chunk_manager_thread, daemon=True),
            threading.Thread(target=self._display_thread, daemon=True),
        ]
        
        for t in threads:
            t.start()
        
        print(f"\nOutput dir: {self.output_dir}")
        print("Running...\n")
        
        try:
            while not STOP_FLAG:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        
        self._cleanup()
    
    def _cleanup(self):
        """Stop capture and release resources."""
        global STOP_FLAG
        STOP_FLAG = True
        self.running = False
        
        print("\n\nStopping...")
        time.sleep(0.5)
        
        # GStreamer
        try:
            if self.pipeline:
                self.pipeline.send_event(Gst.Event.new_eos())
                bus = self.pipeline.get_bus()
                bus.timed_pop_filtered(2 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
                self.pipeline.set_state(Gst.State.NULL)
        except Exception as e:
            print(f"[WARN] GStreamer cleanup error: {e}")
        
        # Close files
        try:
            if self.mocap_file:
                self.mocap_file.close()
            if self.video_ts_file:
                self.video_ts_file.close()
        except:
            pass
        
        print(f"\n{'='*70}")
        print("Capture complete!")
        print(f"Chunks: {self.chunk_index + 1}")
        print(f"Video frames: {self.total_video_frames}")
        print(f"Mocap frames: {self.total_mocap_frames}")
        print(f"Output dir: {self.output_dir}")
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
