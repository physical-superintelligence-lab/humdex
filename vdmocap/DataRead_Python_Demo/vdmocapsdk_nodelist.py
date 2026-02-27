LENGTH_BODY = 23
LENGTH_HAND = 20
LENGTH_FACE = 52
BROADCAST_MAX_NUM = 20
UPDATEID_MAX_ID = 10

NAMES_JOINT_BODY = [
    "Hips",  # 0
    "RightUpperLeg",
    "RightLowerLeg",
    "RightFoot",
    "RightToe",
    "LeftUpperLeg",  # 5
    "LeftLowerLeg",
    "LeftFoot",
    "LeftToe",
    "Spine",
    "Spine1",  # 10
    "Spine2",
    "Spine3",
    "Neck",
    "Head",
    "RightShoulder",  # 15
    "RightUpperArm",
    "RightLowerArm",
    "RightHand",
    "LeftShoulder",
    "LeftUpperArm",  # 20
    "LeftLowerArm",
    "LeftHand",
]

NAMES_JOINT_HAND_RIGHT = [
    "RightHand",
    "RightThumbFinger",
    "RightThumbFinger1",
    "RightThumbFinger2",
    "RightIndexFinger",
    "RightIndexFinger1",
    "RightIndexFinger2",
    "RightIndexFinger3",
    "RightMiddleFinger",
    "RightMiddleFinger1",
    "RightMiddleFinger2",
    "RightMiddleFinger3",
    "RightRingFinger",
    "RightRingFinger1",
    "RightRingFinger2",
    "RightRingFinger3",
    "RightPinkyFinger",
    "RightPinkyFinger1",
    "RightPinkyFinger2",
    "RightPinkyFinger3",
]

NAMES_JOINT_HAND_LEFT = [
    "LeftHand",
    "LeftThumbFinger",
    "LeftThumbFinger1",
    "LeftThumbFinger2",
    "LeftIndexFinger",
    "LeftIndexFinger1",
    "LeftIndexFinger2",
    "LeftIndexFinger3",
    "LeftMiddleFinger",
    "LeftMiddleFinger1",
    "LeftMiddleFinger2",
    "LeftMiddleFinger3",
    "LeftRingFinger",
    "LeftRingFinger1",
    "LeftRingFinger2",
    "LeftRingFinger3",
    "LeftPinkyFinger",
    "LeftPinkyFinger1",
    "LeftPinkyFinger2",
    "LeftPinkyFinger3",
]

NAMES_FACE_EXPRESSION = [
    "BrowDownLeft",  # 0
    "BrowDownRight",
    "BrowInnerUp",
    "BrowOuterUpLeft",
    "BrowOuterUpRight",
    "CheekPuff",
    "CheekSquintLeft",
    "CheekSquintRight",
    "EyeBlinkLeft",
    "EyeBlinkRight",
    "EyeLookDownLeft",  # 10
    "EyeLookDownRight",
    "EyeLookInLeft",
    "EyeLookInRight",
    "EyeLookOutLeft",
    "EyeLookOutRight",
    "EyeLookUpLeft",
    "EyeLookUpRight",
    "EyeSquintLeft",
    "EyeSquintRight",
    "EyeWideLeft",  # 20
    "EyeWideRight",
    "JawForward",
    "JawLeft",
    "JawOpen",
    "JawRight",
    "MouthClose",
    "MouthDimpleLeft",
    "MouthDimpleRight",
    "MouthFrownLeft",
    "MouthFrownRight",  # 30
    "MouthFunnel",
    "MouthLeft",
    "MouthLowerDownLeft",
    "MouthLowerDownRight",
    "MouthPressLeft",
    "MouthPressRight",
    "MouthPucker",
    "MouthRight",
    "MouthRollLower",
    "MouthRollUpper",  # 40
    "MouthShrugLower",
    "MouthShrugUpper",
    "MouthSmileLeft",
    "MouthSmileRight",
    "MouthStretchLeft",
    "MouthStretchRight",
    "MouthUpperUpLeft",
    "MouthUpperUpRight",
    "NoseSneerLeft",
    "NoseSneerRight",  # 50
    "TongueOut",
]

PARENT_INDEXES_BODY = [
    -1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 12, 15, 16, 17, 12, 19,
    20, 21
]

PARENT_INDEXES_HAND = [
    -1, 0, 1, 2, 0, 4, 5, 6, 0, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18
]