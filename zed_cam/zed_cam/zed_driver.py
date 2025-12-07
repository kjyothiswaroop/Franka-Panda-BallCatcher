import queue
import signal
import sys
import threading
import traceback

import cv2
import numpy as np
import pyzed.sl as sl
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header


class ZedDriver(Node):

    def __init__(self):
        super().__init__('zed_driver')

        self.declare_parameter(
            'raw_image_topic',
            'zed/image_raw'
        )

        self.declare_parameter(
            'depth_image_topic',
            'zed/depth_image'
        )

        self.declare_parameter(
            'camera_info_topic',
            'zed/camera_info'
        )

        self.declare_parameter(
            'resolution',
            'HD720'
        )

        self.declare_parameter(
            'fps',
            60
        )

        self.declare_parameter(
            'frame_queue_size',
            1
        )

        self.declare_parameter(
            'camera_frame',
            'zed_camera_link'
        )
       
        self.raw_topic = self.get_parameter('raw_image_topic').value
        self.depth_img_topic = self.get_parameter('depth_image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        res_choice = self.get_parameter('resolution').value
        fps_choice = self.get_parameter('fps').value
        qsz = self.get_parameter('frame_queue_size').value
        self.camera_frame = self.get_parameter('camera_frame').value

        res_map = {
            'HD1080': sl.RESOLUTION.HD1080,
            'HD720': sl.RESOLUTION.HD720,
            'VGA': sl.RESOLUTION.VGA,
            'HD2K': sl.RESOLUTION.HD2K
        }
        self.zed_resolution = res_map.get(res_choice, sl.RESOLUTION.HD1080)
        self.zed_fps = fps_choice
        
        #Publishers # noqa: E26
        qos = rclpy.qos.QoSProfile(depth=10)
        qos.reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT
        qos.durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE

        self.raw_pub = self.create_publisher(
            Image,
            self.raw_topic,
            10
        )

        self.depth_pub = self.create_publisher(
            Image,
            self.depth_img_topic,
            10
        )

        self.cam_info_pub = self.create_publisher(
            CameraInfo,
            self.camera_info_topic,
            10
        )

        self.frame_queue = queue.Queue(maxsize=qsz)

        self.zed = None
        self.mat_rgb = None
        self.mat_depth = None
        self.bridge = CvBridge()

        # Thread control events
        self._stop_event = threading.Event()
        self._grab_thread = None
        self._pub_thread = None

        # Initialize ZED
        try:
            self.init_zed()
        except Exception as e:
            self.get_logger().error(f'ZED initialization failed: {e}')
            self.get_logger().debug(traceback.format_exc())
            raise

        # Start threads
        self._grab_thread = threading.Thread(
            target=self.grab_loop,
            name='zed_grab_thread',
            daemon=True
        )
        self._grab_thread.start()

        self._pub_thread = threading.Thread(
            target=self.publish_loop,
            name='zed_pub_thread',
            daemon=True
        )
        self._pub_thread.start()
        self.get_logger().info('ZED driver node initialized and threads started.')

    def init_zed(self):
        """Initialise Zed Camera."""
        self.zed = sl.Camera()

        init_params = sl.InitParameters()
        init_params.camera_resolution = self.zed_resolution
        init_params.camera_fps = self.zed_fps
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.sdk_verbose = 1

        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f'ZED.open() returned {status}')

        # get resolution and intrinsics
        res = sl.get_resolution(self.zed_resolution)
        self.width = int(res.width)
        self.height = int(res.height)

        cam_info = self.zed.get_camera_information()
        cam_info_struct = cam_info.camera_configuration.calibration_parameters
        left_cam = cam_info_struct.left_cam
        # intrinsics
        self.fx = left_cam.fx
        self.fy = left_cam.fy
        self.cx = left_cam.cx
        self.cy = left_cam.cy

        # distortion parameters (k1,k2,p1,p2,k3)
        try:
            dist = [left_cam.distortion[0], left_cam.distortion[1], left_cam.distortion[2],
                    left_cam.distortion[3], left_cam.distortion[4]]
        except Exception:
            dist = [0.0]*5

        self.dist = dist
        # Prepare sl.Mat containers
        self.mat_rgb = sl.Mat(self.width, self.height, sl.MAT_TYPE.U8_C4)
        self.mat_depth = sl.Mat(self.width, self.height, sl.MAT_TYPE.F32_C1)

        self.get_logger().info(
            f'ZED opened: {self.width}x{self.height} @ {self.zed_fps} fps (requested). fx={self.fx:.2f}, fy={self.fy:.2f}'
        )

    def grab_loop(self):
        """Grab the image using zed.grab() from the ZED SDK."""

        runtime = sl.RuntimeParameters()

        while rclpy.ok() and not self._stop_event.is_set():
            err = self.zed.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(self.mat_rgb, sl.VIEW.LEFT)
                self.zed.retrieve_measure(self.mat_depth, sl.MEASURE.DEPTH)

                raw_image = self.mat_rgb.get_data()
                raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGRA2BGR)

                depth_img = self.mat_depth.get_data()

                zed_ts = self.zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)
                sec = int(zed_ts.get_seconds())
                nanosec = int((zed_ts.get_milliseconds() % 1000) * 1_000_000)

                try:
                    if self.frame_queue.full():
                        try:
                            _ = self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.frame_queue.put_nowait((raw_image, depth_img, sec, nanosec))
                except queue.Full:
                    pass

        self.get_logger().info('Exiting grab loop.')

    def publish_loop(self):
        """Publishes latest frames from the frame_queue as ROS Image messages."""
        while rclpy.ok() and not self._stop_event.is_set():
            try:
                bgr, depth_np, sec, nanosec = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Build header
            header = Header()
            header.stamp.sec = int(sec)
            header.stamp.nanosec = int(nanosec)
            header.frame_id = self.camera_frame

            # CameraInfo per-frame
            cam_info_msg = CameraInfo()
            cam_info_msg.header = header
            cam_info_msg.width = self.width
            cam_info_msg.height = self.height
            cam_info_msg.k = [
                self.fx, 0.0, self.cx,
                0.0, self.fy, self.cy,
                0.0, 0.0, 1.0
            ]
            cam_info_msg.p = [
                self.fx, 0.0, self.cx, 0.0,
                0.0, self.fy, self.cy, 0.0,
                0.0, 0.0, 1.0, 0.0
            ]
            cam_info_msg.distortion_model = 'plumb_bob'
            cam_info_msg.d = self.dist
            self.cam_info_pub.publish(cam_info_msg)

            # Color image (bgr8)
            try:
                color_msg = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
                color_msg.header = header
                self.raw_pub.publish(color_msg)
            except Exception as e:
                self.get_logger().warn(f'Failed to publish color image: {e}')

            # Depth image (32FC1, meters)
            try:
                if depth_np.dtype != np.float32:
                    depth_np = depth_np.astype(np.float32)
                depth_msg = self.bridge.cv2_to_imgmsg(depth_np, encoding='32FC1')
                depth_msg.header = header
                self.depth_pub.publish(depth_msg)
            except Exception as e:
                self.get_logger().warn(f'Failed to publish depth image: {e}')

        self.get_logger().info('Exiting publish loop.')
    
    def destroy_node(self):
        self.get_logger().info('Shutting down ZED driver node...')
        self._stop_event.set()

        if self._grab_thread is not None:
            self._grab_thread.join(timeout=1.0)
        if self._pub_thread is not None:
            self._pub_thread.join(timeout=1.0)

        try:
            if self.zed is not None:
                self.zed.close()
        except Exception as e:
            self.get_logger().warn(f'Error closing ZED: {e}')

        super().destroy_node()

def main(args=None):
    """Main entry point for node."""
    def sigint_handler(sig, frame):
        pass
    signal.signal(signal.SIGINT, sigint_handler)

    rclpy.init(args=args)
    node = ZedDriver()
    executor = MultiThreadedExecutor(num_threads=4)
    try:
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()
