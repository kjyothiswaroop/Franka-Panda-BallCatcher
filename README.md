## ME450-FinalProject-Catchers (Group 1)

This repository holds the package for the ME-450 final project of Team Catchers.

## Prerequisites
- The `realsense2_camera` driver installed (from Intel RealSense ROS2 package) if using a RealSense camera.
- A RealSense camera physically connected to the machine and accessible by the driver.
- Clone and build the `robot_mover` package:
    ```bash
    mkdir -p mover_ws/src
    cd mover_ws/src
    git clone git@github.com:ME495-EmbeddedSystems/homework-3-part-2-actual-catchers.git
    cd ../
    colcon build
    ```
- Clone and build the `easy_handeye2` package :
    ```bash
    mkdir -p handeye_ws/src
    cd handeye_ws/src
    git clone git@github.com:kjyothiswaroop/easy_handeye2.git
    cd ../
    colcon build
    ```
- Environment setup:
   Add the lines below to your ~/.bashrc and replace `<location_of_mover_ws>` and `<location_of_handeye_ws>` with the correct locations on your computer.
   ```bash
   source ~/<location_of_mover_ws>/mover_ws/install/setup.bash
   source ~/<location_of_handeye_ws>/handeye_ws/install/setup.bash
   ```

- Clone and build the `catchers_vision` and `catchers_vision_interfaces` packages:
  ```bash
  mkdir -p catchers_ws/src
  cd catchers_ws/src
  git clone git@github.com:ME495-EmbeddedSystems/final-project-catchers.git
  cd ../
  colcon build
  source install/setup.bash
  ```

## Quickstart
The `catchers_vision` package has one important launch file `detect.launch.xml.`
- It expects two parameters which are :
    - `demo` (default value = true)
    - `calibrate` (default value = false)

- ### What `detect.launch.xml` does

    - Starts the `rs_launch.py` from the `realsense2_camera` package.
        - Camera is launched at 60fps at a lower resolution and depth alignment.

    - Starts the `pickplacelaunch.xml` from `robot_mover` package.

        - If `demo` is `true` then the franka launches in Rviz along with Moveit.

        - If `demo` is `false` then only the rviz launches(it is expected that a real Franka arm is connected and running the moveit launch file on the real Franka arm.)

    - If `calibrate` is `true`:

        - The `aruco_detect` node starts and : 
            - Loads parameters from `config/aruco_params.yaml` (marker size, dictionary, detection thresholds, etc.).

            - Subscribes to the camera image and camera info topics exposed by a camera driver (for this project we use an Intel RealSense camera).

            - Performs ArUco marker detection and publishes marker info (see `catchers_vision_interfaces/msg/ArucoMarkers.msg`) and publishes updated image with ArUco Detection on a new ROS2 topic.

        - The `calibrate.launch.xml` :
            - Launches the calibration pipeline of `easy_handeye2` package.

            - At the end of the pipeline, saves the calibration to `.ros2/easy_handeye2/calibrations/my_eob_calib.calib`.

    - If calibrate is `false`:
        
        - The `publish.launch.xml` :
            - Launches the transform publishing pipeline from the calibration step.

            - Publishes a transform between `base` and `camera_link` frames.
        
        - The `ball_track` node from the `catchers_vision` package:
            - Detects and tracks a ball in 3D space using OpenCV or YOLO models.

            - Publishes a transform between `camera_color_optical_frame` and `ball` frames.

            - Publishes the detected ball in the image with the centroid on a new ROS2 topic.



## How to run

1. Plug in the RealSense camera into your machine.

2. In another terminal (with the catchers_ws sourced), launch the `detect.launch.xml`:

- If camera calibration is required(usually when camera is moved), attach the ArUco marker of ID `25` from the family
  `DICT_6X6_1000` at the end effector of the franka robot arm and run the command below and follow the calibration procedure as described in [Camera_Calibration](https://github.com/kjyothiswaroop/easy_handeye2/blob/master/README.md)

    ```bash
    ros2 launch catchers_vision detect.launch.xml demo:=false calibrate:=true
    ```

- If calibration file already exists at `.ros2/easy_handeye2/calibrations/my_eob_calib.calib`, then run the below command to start detecting the ball and publishing transforms.

    ```bash
    ros2 launch catchers_vision detect.launch.xml demo:=false calibrate:=false
    ```

## Directory Structure

```bash
final-project-catchers
│
├── catchers_vision
│   ├── catchers_vision
│   │   ├── aruco.py
│   │   ├── ball_track.py
│   │   ├── cv.py
│   │   ├── __init__.py
│   │   ├── stream.py
│   │   ├── test_stream.py
│   │   ├── trajectory_prediction.py
│   │   └── traj_pred_node.py
│   ├── config
│   │   └── aruco_params.yaml
│   ├── launch
│   │   ├── calib.launch.xml
│   │   ├── detect.launch.xml
│   │   └── publish.launch.xml
│   ├── package.xml
│   ├── resource
│   │   └── catchers_vision
│   ├── setup.cfg
│   ├── setup.py
│   └── test
│       ├── test_copyright.py
│       ├── test_flake8.py
│       ├── test_pep257.py
│       └── test_xmllint.py
├── catchers_vision_interfaces
│   ├── CMakeLists.txt
│   ├── LICENSE
│   ├── msg
│   │   └── ArucoMarkers.msg
│   └── package.xml
├── final.repos
└── README.md
```
