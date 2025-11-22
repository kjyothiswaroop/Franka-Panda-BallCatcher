from enum import auto, Enum

from geometry_msgs.msg import PointStamped, TransformStamped

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from std_srvs.srv import Empty

from tf2_ros import TransformBroadcaster

from . import stream


class VisionState(Enum):
    """Current style used to track ball."""

    OPENCV = auto()
    YOLO = auto()
    HAAR = auto()


def color_threshold(stream, ball):
    """Check frame for ball and publishes if present."""
    stream.set_scale()
    stream.align_self()
    while True:
        frame = stream.capture_frame()
        frame_HSV = stream.convert_color(frame)
        frame_green, mask = stream.threshold_ball(frame_HSV, ball)
        (cx, cy, cz), pnt = stream.find_ball(mask)
        if cx != -1:
            return np.array([cx, cy, cz])
        else:
            return np.array([-1, -1, -1])


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

        self.mode = (
            self.get_parameter('mode').get_parameter_value().string_value
        )
        self.ball = (
            self.get_parameter('ball_type').get_parameter_value().string_value
        )

        self._ball = self.create_publisher(
            PointStamped, '/ball_pose', qos_profile
        )

        self._track = self.create_service(Empty, '/track', self.track_callback)
        self.state = VisionState.OPENCV

    def track_callback(self, request, response):
        """Activates ball tracking."""
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
                    transform = TransformStamped()
                    transform.header.stamp = self.get_clock().now().to_msg()
                    transform.header.frame_id = 'camera'
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
