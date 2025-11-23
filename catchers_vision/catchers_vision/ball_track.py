import cv2
from cv_bridge import CvBridge

from enum import auto, Enum

from geometry_msgs.msg import PointStamped, TransformStamped

from message_filters import Subscriber, ApproximateTimeSynchronizer

import numpy as np

import pyrealsense2 as rs

from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from sensor_msgs.msg import Image, CameraInfo

from std_srvs.srv import Empty

from tf2_ros import TransformBroadcaster

from . import stream

lower_tennis = np.array([15, 65, 50])
upper_tennis = np.array([35, 255, 200])

lower_green = np.array([50, 150, 0])
upper_green = np.array([80, 255, 170])

# Not a great color threshold
lower_red = np.array([0, 30, 30])
upper_red = np.array([3, 255, 210])

lower_orange = np.array([0, 185, 100])
upper_orange = np.array([15, 255, 255])


class VisionState(Enum):
    """Current style used to track ball."""

    OPENCV = auto()
    YOLO = auto()
    HAAR = auto()


def convert_color(frame):
    """Convert color space from bgr to hsv."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return hsv


def threshold_ball(image, frame, ball):
    """Isolate ball."""
    if ball == 'tennis':
        mask = cv2.inRange(frame, lower_tennis, upper_tennis)
    elif ball == 'green':
        mask = cv2.inRange(frame, lower_green, upper_green)
    elif ball == 'red':
        mask = cv2.inRange(frame, lower_red, upper_red)
    else:
        mask = cv2.inRange(frame, lower_orange, upper_orange)

    return cv2.bitwise_and(
        image, image, mask=mask
    ), mask


def color_threshold(color_img, depth_img, intr, ball):
    """Check frame for ball and publishes if present."""
    # stream.set_scale()
    # stream.align_self()
    while True:
        frame_HSV = convert_color(color_img)
        frame_green, mask = threshold_ball(frame_HSV, ball)
        (cx, cy, cz), pnt = find_ball(mask, depth_img, intr)
        if cx != -1:
            return np.array([float(cx), float(cy), float(cz)])
        else:
            return np.array([-1.0, -1.0, -1.0])


class BallTrack(Node):
    """Publish pose of ball that node is attempting to track."""

    def __init__(self):
        """Initialize the ball tracking node."""
        super().__init__('ball_track')
        # Establish Broadcasters:
        self.broadcaster = TransformBroadcaster(self)

        qos_profile = QoSProfile(depth=10)

        self.declare_parameter('mode', 'open_cv')
        self.declare_parameter('ball_type', 'green')
        self.declare_parameter(
            'image_topic',
            '/camera/image_raw',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )

        self.mode = (
            self.get_parameter('mode').get_parameter_value().string_value
        )
        self.ball = (
            self.get_parameter('ball_type').get_parameter_value().string_value
        )
        self.image_topic = self.get_parameter('image_topic').value

        # INLINE CITATION ###
        self.color_sub = Subscriber(self, Image, "/camera/color/image_raw")
        self.depth_sub = Subscriber(self, Image, "/camera/aligned_depth_to_color/image_raw")
        self.create_subscription(
            CameraInfo,
            "/camera/color/camera_info",
            self.camera_info_callback,
            10
        )
        self.intrinsics = None
        self.got_intrinsics = False
        
        self.ts = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=10,
            slop=0.05  # 50 ms tolerance
        )
        self.ts.registerCallback(self.synced_callback)

        self._ball = self.create_publisher(
            PointStamped, '/ball_pose', qos_profile
        )
        self.image_pub = self.create_publisher(Image, self.image_topic + '_axes', 10)

        # self._track = self.create_service(Empty, '/track', self.track_callback)
        self.state = VisionState.OPENCV
        self.bridge = CvBridge()

    def synced_callback(self, color_msg, depth_msg):
        """Activates ball tracking."""
        #TODO: Figure out a way to remove the stream thing, because it creates issues with tf tree.
        #Assume that camera gives you images on a topic and process that into the stream function.
        if self.state == VisionState.OPENCV:
            with stream.Stream() as f:
                while True:
                    location = color_threshold(f, self.ball)
                    pt = PointStamped()
                    pt.header.stamp = self.get_clock().now().to_msg()
                    pt.point.x = location[0]
                    pt.point.y = location[1]
                    pt.point.z = location[2]
                    self._ball.publish(pt)
                    if pt.point.z != -1.0:
                        transform = TransformStamped()
                        transform.header.stamp = self.get_clock().now().to_msg()
                        transform.header.frame_id = 'camera_color_optical_frame'
                        transform.child_frame_id = 'ball'

                        transform.transform.translation.x = location[0]
                        transform.transform.translation.y = location[1]
                        transform.transform.translation.z = location[2]
                        transform.transform.rotation.w = 1.0
                        self.broadcaster.sendTransform(transform)


def main(args=None):
    """Entry point for the arena node."""
    rclpy.init(args=args)
    node = BallTrack()
    rclpy.spin(node)
    rclpy.shutdown()
