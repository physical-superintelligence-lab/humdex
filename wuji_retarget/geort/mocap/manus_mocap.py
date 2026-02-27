# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import zmq
import numpy as np
import threading 

class ManusMocap:
    '''
    Applies to any ZMQ-broadcasted mocap data with fixed shape (21,3) and dtype float32.
    Runs a background thread to continuously receive and update latest data.
    '''
    def __init__(self, port=8765):
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(f"tcp://localhost:{port}")
        socket.setsockopt_string(zmq.SUBSCRIBE, "") 
        self.socket = socket

        self._latest_data = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self):
        while self._running:
            try:
                msg = self.socket.recv(flags=zmq.NOBLOCK)
                arr = np.frombuffer(msg, dtype=np.float32).reshape(21, 3)
                with self._lock:
                    self._latest_data = arr
            except zmq.Again:
                import time
                time.sleep(0.001)

    def get(self):
        with self._lock:
            if self._latest_data is not None:
                return {"result": self._latest_data.copy(), "status": "recording"}
            else:
                return {"result": None, "status": "no data"}

    def close(self):
        self._running = False
        self._thread.join()
        self.socket.close()