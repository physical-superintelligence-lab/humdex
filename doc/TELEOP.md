# Teleop Pipeline


1.1 Start the G1 robot
1.2 `bash docker_neck.sh` for starting neck server
1.3 replug the ZED MINI camera to ensure connection
1.4 `bash docker_zed.sh` for starting orin zed sender
1.5 Start to listen to ZED MINI camera in VR (you should now see the camera feed in the VR)


2.1 Wear motion trackers; wear controllers on the wrists
2.2 Start the VR
2.3 Calibrate the whole-body motions
2.4 Enter XRobot APP
2.5 Connect to IP of my ubuntu
2.6 Start to streaming whole-body data and hand data
2.7 start teleop in mujoco
```bash
bash teleop_motion_gen.sh
```
2.8 test sim2sim first
```bash
bash sim2sim.sh
```



3.1 Start data recording
```bash
bash teleop_data_record.sh
```

3.2 start sim2real
```bash
bash sim2real.sh
```

# 按键说明

teleop:

    右手A: 开始/暂停teleop
    左手X: 退出teleop,进入default pose

    右手index grip: close right hand
    右手grip: open right hand

    左手index grip: close left hand
    左手grip: open left hand

    右手B: 缩小streamed RGB in VR

    左手axis click: robot急停
    

motion gen:

    右手axis clik: 进入/退出motion gen mode

    左手方向盘: motion gen mode下，控制机器人运动x
    右手方向盘: motion gen mode下，控制机器人运动yaw

data record:

    左手Y: 开始/暂停 data record

    左手axis click: 退出data record




# APP Setting

Pull from PICO:
```bash
adb pull /sdcard/Android/data/com.xrobotoolkit.client/files/video_source.yml video_source.yml
```

Push to PICO:
```bash
adb push video_source.yml /sdcard/Android/data/com.xrobotoolkit.client/files/video_source.yml
```

