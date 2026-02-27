# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Meta Internal Manus. 
import math
import threading
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import zmq 


def hand_to_canonical(hand_point):
    z_axis = hand_point[9] - hand_point[0]
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis_aux = hand_point[5] - hand_point[13]
    y_axis_aux = y_axis_aux / np.linalg.norm(y_axis_aux)

    x_axis = np.cross(y_axis_aux, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    rotation_base = np.array([x_axis, y_axis, z_axis]).transpose()
    tranlation_base = hand_point[0]

    transform = np.eye(4)
    transform[:3, :3] = rotation_base
    transform[:3, 3] = tranlation_base
    
    transform_inv = np.linalg.inv(transform)
    hand_point = np.array(hand_point)
    hand_point = np.concatenate((np.array(hand_point), np.ones((21, 1))), axis=-1)
    hand_point = hand_point @ transform_inv.transpose()
    return hand_point[:, :3]



class ManusForwardKinematicsSolver:
    def __init__(self):
        return 

    def make_transformation_matrix(self, pos, quat):
        from scipy.spatial.transform import Rotation as R
        out = np.eye(4)
        out[:3, 3] = pos
        out[:3, :3] = R.from_quat(quat).as_matrix()
        return out

    def solve_keypoints(self, positions, orientation):
        thumb_chain = [0, 1, 2, 3, 4]
        index_chain = [0, 5, 6, 7, 8]
        middle_chain = [0, 9, 10, 11, 12]
        ring_chain = [0, 13, 14, 15, 16]
        pinky_chain = [0, 17, 18, 19, 20]
        all_chains = [thumb_chain, index_chain, middle_chain, ring_chain, pinky_chain]

        all_keypoints = {}
        
        for chain in all_chains:
            current_transformation_to_world = np.eye(4)

            for idx in chain:
                pos = np.array(positions[idx])
                transformation = self.make_transformation_matrix(pos, orientation[idx])

                last_position = np.array(current_transformation_to_world[:3, 3])
                current_transformation_to_world = current_transformation_to_world @ transformation
                position = current_transformation_to_world[:3, 3]

                if idx not in all_keypoints:
                    all_keypoints[idx] = position

        return all_keypoints

class Manus(Node):
    def __init__(self):
        super().__init__("manus_visualizer")
        self.x_axis = []
        self.y_axis = []
        self.z_axis = []
        self.pos = None
        self.quat = None

        # Damn, this part is manually measured human finger link vectors.
        self.pos = np.array([
            [0.0, 0.0, 0.0],
            [0.0250, 0.0000, 0.0050],
            [0.0000, 0.0000, 0.0390],
            [0.0000, 0.0000, 0.0330],
            [0.0000, 0.0000, 0.0210],
            [0.0170, 0.0000, 0.0870],
            [0.0000, 0.0000, 0.0260],
            [0.0000, 0.0000, 0.0220],
            [0.0000, 0.0000, 0.0200],
            [0.0000, 0.0000, 0.0920],
            [0.0000, 0.0000, 0.0260],
            [0.0000, 0.0000, 0.0260],
            [0.0000, 0.0000, 0.0220],
            [-0.0170, 0.0000, 0.0840],
            [0.0000, 0.0000, 0.0210],
            [0.0000, 0.0000, 0.0210],
            [0.0000, 0.0000, 0.0200],
            [-0.0340, 0.0000, 0.0720],
            [0.0000, 0.0000, 0.0210],
            [0.0000, 0.0000, 0.0210],
            [0.0000, 0.0000, 0.0200],
        ])

        self.manus_x_subscription = self.create_subscription(
            Float32MultiArray, "/x_manus_rotations", self.listener_callback_x, 10
        )

        self.manus_y_subscription = self.create_subscription(
            Float32MultiArray, "/y_manus_rotations", self.listener_callback_y, 10
        )

        self.manus_z_subscription = self.create_subscription(
            Float32MultiArray, "/z_manus_rotations", self.listener_callback_z, 10
        )

        self.manus_quats_subscription = self.create_subscription(
            Float32MultiArray, "/manus_quats", self.listener_callback_quat, 10
        )

        # Broadcast on localhost
        self.port = 8765
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.port}")
        self.socket.setsockopt(zmq.SNDHWM, 0)

    def listener_callback_x(self, msg):
        self.x_msg = list(msg.data)
        self.x_axis = [math.degrees(r) for r in self.x_msg]

    def listener_callback_y(self, msg):
        self.y_msg = list(msg.data)
        self.y_axis = [math.degrees(r) for r in self.y_msg]

    def listener_callback_z(self, msg):
        self.z_msg = list(msg.data)
        self.z_axis = [math.degrees(r) for r in self.z_msg]

    def listener_callback_quat(self, msg):
        self.quat = np.array(list(msg.data)).reshape(21, 4)

    def run(self):
        count = 0
        kinematics_solver = ManusForwardKinematicsSolver()
        while rclpy.ok():
            count = count + 1
            if self.pos is None or self.quat is None or len(self.x_axis) == 0:
                continue

            keypoints = kinematics_solver.solve_keypoints(self.pos, self.quat)
            keypoints =  np.array([keypoints[i] for i in range(21)])
            keypoints = hand_to_canonical(keypoints).astype(np.float32)

            self.socket.send(keypoints.tobytes())
            print("Broadcasting Manus Reading", count, keypoints.shape)
     
def main(args=None):
    rclpy.init(args=args)
    manus_node = Manus()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(manus_node)
    run_thread = threading.Thread(target=manus_node.run)
    run_thread.start()
    executor.spin()
    run_thread.join()  


if __name__ == "__main__":
    main()
