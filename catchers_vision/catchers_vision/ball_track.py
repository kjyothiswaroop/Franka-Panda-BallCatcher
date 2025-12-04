from enum import auto, Enum

from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, TransformStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster
from ultralytics import YOLO

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

        self.get_logger().info('ball_track')
        qos_profile = QoSProfile(depth=10)

        #Parameter declaration. # noqa: E26
        self.declare_parameter('mode', 'yolo')
        self.declare_parameter('ball_type', 'orange')
        self.declare_parameter(
            'image_topic',
            '/camera/image_raw',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        self.declare_parameter('model', value='ball-detect-3.pt')

        #Read param values. # noqa: E26
        self.mode = (
            self.get_parameter('mode').get_parameter_value().string_value
        )
        self.ball = (
            self.get_parameter('ball_type').get_parameter_value().string_value
        )
        self.model = YOLO(self.get_parameter('model').get_parameter_value().string_value)
        self.image_topic = self.get_parameter('image_topic').value

        self.color_img = Image()
        self.depth_img = Image()
        if self.mode == 'yolo':
            self.state = VisionState.YOLO
        else:
            self.state = VisionState.OPENCV
        self.bridge = CvBridge()
        self.img_proc = cv.image_processor()
        #Subscribers # noqa: E26
        self.color_sub = Subscriber(
            self,
            Image,
            '/camera/camera/color/image_raw'
        )
        self.depth_sub = Subscriber(
            self,
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw'
        )
        self.caminfo_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        self.ts = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=10,
            slop=0.05  # 50 ms tolerance
        )
        self.ts.registerCallback(self.synced_callback)

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
        self.timer = self.create_timer(0.01, self.timer_callback)

        #Attributes # noqa: E26
        self.intrinsics = None
        self.got_intrinsics = False
        self.color_img = None
        self.depth_img = None
        # self.state = VisionState.OPENCV
        self.bridge = CvBridge()
        self.img_proc = cv.image_processor()
        self.points = []

    def timer_callback(self):
        """Activates ball tracking."""
        if self.state == VisionState.OPENCV:
            # self.get_logger().info('OpenCV mode')
            if self.got_intrinsics:

                if self.color_img is None or self.depth_img is None:
                    self.get_logger().warn('Waiting for images...')
                    return

                color_img = self.bridge.imgmsg_to_cv2(
                    self.color_img,
                    desired_encoding='bgr8'
                )

                depth_img = self.bridge.imgmsg_to_cv2(
                    self.depth_img,
                )

                location, mask = self.img_proc.color_threshold(
                    color_img,
                    depth_img,
                    self.intrinsics,
                    self.ball
                )

                mask_msg = self.bridge.cv2_to_imgmsg(mask)
                self.image_pub.publish(mask_msg)

                self.broadcast_ball(location)
        elif self.state == VisionState.YOLO:
            # self.get_logger().info('YOLO Model Mode')
            if self.got_intrinsics:
                color_img = self.bridge.imgmsg_to_cv2(
                    self.color_img,
                    desired_encoding='bgr8'
                )

                depth_img = self.bridge.imgmsg_to_cv2(
                    self.depth_img,
                )

                results = self.model(color_img)
                class_names = self.model.names
                # Get the result and draw it on an OpenCV image
                frame = results[0].plot()
                cx, cy = self.img_proc.yolo_find_ball(results[0], class_names)
                location = self.img_proc.depth_extract(cx, cy, depth_img, self.intrinsics)
                yolo_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                self.image_pub.publish(yolo_msg)

                self.broadcast_ball(location)

    def broadcast_ball(self, location):
        """Broadcast ball tf to tf tree."""
        if location[2] != -1.0:
            self.get_logger().info(f'ball detected at {location}')
        else:
            self.get_logger().info('Ball not detected!')

        pt = PointStamped()
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.point.x = location[0]
        pt.point.y = location[1]
        pt.point.z = location[2]
        self._ball.publish(pt)

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'camera_color_optical_frame'
        transform.child_frame_id = 'ball'
        transform.transform.translation.x = location[0]
        transform.transform.translation.y = location[1] - 0.05
        transform.transform.translation.z = location[2]
        transform.transform.rotation.w = 1.0
        self.broadcaster.sendTransform(transform)

    def camera_info_callback(self, msg):
        """Camera info callback."""
        fx = msg.k[0]
        fy = msg.k[4]
        cx = msg.k[2]
        cy = msg.k[5]

        if not self.got_intrinsics:
            self.intrinsics = (fx, fy, cx, cy)
            self.got_intrinsics = True

    def synced_callback(self, color_msg, depth_msg):
        """Sync callback for color and depth."""
        self.color_img = color_msg
        self.depth_img = depth_msg


def main(args=None):
    """Entry point for the ball_track node."""
    rclpy.init(args=args)
    node = BallTrack()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    import sys
    main(sys.argv)
