import cv2
from cv_bridge import CvBridge

from enum import auto, Enum

from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, TransformStamped
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster

from . import cv


class VisionState(Enum):
    """Current style used to track ball."""

    OPENCV = auto()
    YOLO = auto()
    HAAR = auto()


class BallTrack(Node):
    """Publish pose of ball that node is attempting to track."""

    def __init__(self):
        """Initialize the ball tracking node."""
        super().__init__('ball_track')
        # Establish Broadcasters:
        self.broadcaster = TransformBroadcaster(self)

        qos_profile = QoSProfile(depth=10)

        #Parameter declaration. # noqa: E26
        self.declare_parameter('mode', 'open_cv')
        self.declare_parameter('ball_type', 'green')
        self.declare_parameter(
            'image_topic',
            '/camera/image_raw',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        #Read param values. # noqa: E26
        self.mode = (
            self.get_parameter('mode').get_parameter_value().string_value
        )
        self.ball = (
            self.get_parameter('ball_type').get_parameter_value().string_value
        )
        self.image_topic = self.get_parameter('image_topic').value
        self.intrinsics = None
        self.got_intrinsics = False

        self.color_img = Image()
        self.depth_img = Image()
        self.state = VisionState.OPENCV
        self.bridge = CvBridge()
        self.img_proc = cv.image_processor()
        #Subscribers # noqa: E26
        self.color_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.color_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )
        self.caminfo_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        #Publishers # noqa: E26
        self._ball = self.create_publisher(
            PointStamped, '/ball_pose', qos_profile
        )
        self.image_pub = self.create_publisher(
            Image,
            self.image_topic + '_axes',
            10
        )

        #Timer callback # noqa: E26
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        """Activates ball tracking."""
        if self.state == VisionState.OPENCV:
            if self.got_intrinsics:
                color_img = self.bridge.imgmsg_to_cv2(
                    self.color_img,
                    desired_encoding='bgr8'
                )

                depth_img = self.bridge.imgmsg_to_cv2(
                    self.depth_img,
                    desired_encoding='passthrough'
                )

                location, mask = self.img_proc.color_threshold(
                    color_img,
                    depth_img,
                    self.intrinsics,
                    self.ball
                )

                mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
                self.image_pub.publish(mask_msg)

                if location[2] != -1.0:
                    self.get_logger().info(f'ball detected at {location}')

                    pt = PointStamped()
                    pt.header.stamp = self.get_clock().now().to_msg()
                    pt.point.x = location[0]
                    pt.point.y = location[1]
                    pt.point.z = location[2]
                    self._ball.publish(pt)
                    transform = TransformStamped()
                    transform.header.stamp = self.get_clock().now().to_msg()
                    transform.header.frame_id = 'camera_link'
                    transform.child_frame_id = 'ball'

                    transform.transform.translation.x = location[0]
                    transform.transform.translation.y = location[1]
                    transform.transform.translation.z = location[2]
                    transform.transform.rotation.w = 1.0
                    self.broadcaster.sendTransform(transform)

                else:
                    self.get_logger().info('Ball not detected!')

    def camera_info_callback(self, msg):
        """Camera info callback."""
        fx = msg.k[0]
        fy = msg.k[4]
        cx = msg.k[2]
        cy = msg.k[5]

        if not self.got_intrinsics:
            self.intrinsics = (fx, fy, cx, cy)
            self.got_intrinsics = True

    def color_callback(self, msg):
        """Color Image callback."""
        self.color_img = msg

    def depth_callback(self, msg):
        """Depth Image callback."""
        self.depth_img = msg


def main(args=None):
    """Entry point for the arena node."""
    rclpy.init(args=args)
    node = BallTrack()
    rclpy.spin(node)
    rclpy.shutdown()
