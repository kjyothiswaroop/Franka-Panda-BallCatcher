## ME450-FinalProject-Catchers (Group 1)

This repository contains the vision package used for the ME450 final project. The `catchers_vision` package includes an ArUco detection node and a launch file `launch/aruco.launch.xml` that starts the ArUco detection stack.

## What `aruco.launch.xml` does

- Starts the ArUco detection node (in the `catchers_vision` package).(For OpenCV>4.7)
- Loads parameters from `config/aruco_params.yaml` (marker size, dictionary, detection thresholds, etc.).
- Subscribes to the camera image and camera info topics exposed by a camera driver (for this project we use an Intel RealSense camera).
- Performs ArUco marker detection and publishes marker info (see `catchers_vision_interfaces/msg/ArucoMarkers.msg`) and any visualization you may have implemented.

## Prerequisites

- ROS 2 (the distro used for the project — make sure your environment is sourced).
- The `realsense2_camera` driver installed (from Intel RealSense ROS2 package) if using a RealSense camera.
- A RealSense camera physically connected to the machine and accessible by the driver.

Note: the package name for the launch command is `catchers_vision`.

## How to run

1. Plug in the Intel RealSense camera into your machine.

2. In a terminal, start the RealSense camera driver so camera topics are available:

```bash
ros2 launch realsense2_camera rs_launch.py
```

Keep that process running (it provides the camera image and camera_info topics the ArUco node expects).

3. In another terminal (with your ROS 2 workspace sourced), launch the ArUco stack:

```bash
ros2 launch catchers_vision aruco.launch.xml
```

This will start the ArUco node with parameters from `catchers_vision/config/aruco_params.yaml` and begin publishing detected marker information.

## Notes and troubleshooting

- If you do not see detected markers, verify the RealSense topics are active (e.g., using `ros2 topic list`) and that the camera frames are correct.
- Check the parameters in `config/aruco_params.yaml` (marker size and dictionary must match the markers in your scene).
- If you changed topic names, or use a different camera driver, update the launch file or node parameters accordingly.
