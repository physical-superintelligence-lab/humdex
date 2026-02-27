# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# ask one of the existing GPTs to adapt the realsense.
# and they will give you this.
import cv2

class WebcamCamera:
    def __init__(self, device_index=0):
        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open webcam (device index {device_index})")

        # Get default resolution from the webcam
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30  # fallback to 30 if FPS not detected

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return {"rgb": None, "depth": None}
        return {"rgb": frame, "depth": None}  # No depth for USB webcam

    def release(self):
        self.cap.release()

    def __del__(self):
        self.release()


def test():
    camera = WebcamCamera()

    try:
        while True:
            frame = camera.get_frame()
            color_frame = frame["rgb"]

            if color_frame is not None:
                cv2.imshow('Color Frame', color_frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    test()