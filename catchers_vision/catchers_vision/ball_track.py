from enum import auto, Enum

from geometry_msgs.msg import PointStamped

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from std_srvs.srv import Empty

from . import stream


class VisionState(Enum):
    """Current style used to track ball."""

    OPENCV = auto()
    YOLO = auto()
    HAAR = auto()


def big_boss(stream):
    """Check frame for ball and publishes if present."""
    stream.set_scale()
    stream.align_self()
    while True:
        frame = stream.capture_frame()
        frame_HSV = stream.convert_color(frame)
        frame_green, mask = stream.threshold_green(frame_HSV)
        (cx, cy, cz), pnt = stream.find_ball(mask)
        if cx != -1:
            return np.array([cx, cy, cz])
        else:
            return None


class BallTrack(Node):
    """Publish pose of ball that node is attempting to track."""

    def __init__(self):
        """Initialize the ball tracking node."""
        super().__init__('ball_track')
        self.get_logger().info('ball_track')
        qos_profile = QoSProfile(depth=10)

        self.declare_parameter('mode', 'open_cv')

        self.mode = (
            self.get_parameter('mode').get_parameter_value().string_value
        )

        self.ball = self.create_publisher(
            PointStamped, '/ball_pose', qos_profile
        )

        self._track = self.create_service(Empty, '/track', self.track_callback)

    def track_callback(self, request, response):
        """Activates ball tracking."""
        with stream.Stream() as f:
            while True:
                location = big_boss(f)
                if location is not None:
                    pt = PointStamped()
                    pt.header.stamp = self.get_clock().now().to_msg()
                    pt.point.x = location[0]
                    pt.point.y = location[1]
                    pt.point.z = location[2]
                    self.ball.publish(pt)

def main(args=None):
    """Entrypoint for pick_node."""
    rclpy.init(args=args)
    node = BallTrack()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    import sys
    main(sys.argv)