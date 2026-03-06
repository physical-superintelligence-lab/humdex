# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pyrealsense2 as rs
import numpy as np
import cv2

class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable both color and depth streams
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        # Start streaming
        self.pipeline.start(self.config)
        
    def get_frame(self):
        try:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None
                
            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            return {"rgb": color_image, "depth": depth_image}
            
        except Exception as e:
            print(f"Error getting frames: {e}")
            return {"rgb": None, "depth": None}
    
    def release(self):
        """Stop and release the camera resources."""
        self.pipeline.stop()
    
    def __del__(self):
        """Destructor to ensure camera is properly released."""
        self.release()


def test():
    camera = RealSenseCamera()
    
    try:
        while True:
            frame = camera.get_frame()
            color_frame = frame["rgb"]
            depth_frame = frame["depth"]
            if color_frame is not None:
                # Show color image
                cv2.imshow('Color Frame', color_frame)
                
                # Apply colormap to depth image for visualization
                if depth_frame is not None:
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_frame, alpha=0.03), 
                        cv2.COLORMAP_JET
                    )
                    cv2.imshow('Depth Frame', depth_colormap)
            
            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        camera.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    test()
   