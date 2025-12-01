from enum import auto, Enum

from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, TransformStamped
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
        self.declare_parameter('model', value='ball_detect.pt')

        #Read param values. # noqa: E26
        self.mode = (
            self.get_parameter('mode').get_parameter_value().string_value
        )
        self.ball = (
            self.get_parameter('ball_type').get_parameter_value().string_value
        )
        self.model = YOLO(self.get_parameter('model').get_parameter_value().string_value)
        self.image_topic = self.get_parameter('image_topic').value
        self.intrinsics = None
        self.got_intrinsics = False

        self.color_img = Image()
        self.depth_img = Image()
        if self.mode == 'yolo':
            self.state = VisionState.YOLO
        else:
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
                cx, cy = yolo_find_ball(results[0], class_names)
                location = depth_extract(cx, cy, depth_img, self.intrinsics)                            
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

    def color_callback(self, msg):
        """Color Image callback."""
        self.color_img = msg

    def depth_callback(self, msg):
        """Depth Image callback."""
        self.depth_img = msg


def yolo_find_ball(results, class_names):
    """Utilize yolo to return location of moving ball if found."""
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = class_names[cls_id]

            if cls_name == 'Moving Ball':
                xywh = box.xywh[0].cpu().numpy().astype(int)
                cx, cy, w, h = xywh
                conf = float(box.conf[0])
                if conf > 0.70:
                    return cx, cy
        return None, None
    else:
        return None, None


def depth_extract(cx, cy, depth_img, intr):
    """Extract depth from points and depth img."""
    if cx is not None:
        depth = depth_img[cy, cx] * 0.001
        fx, fy, cx0, cy0 = intr
        X = (cx - cx0) * depth / fx
        Y = (cy - cy0) * depth / fy
        Z = depth
        return [X, Y, Z]
    else:
        return [-1.0, -1.0, -1.0]


def main(args=None):
    """Entry point for the ball_track node."""
    rclpy.init(args=args)
    node = BallTrack()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    import sys
    main(sys.argv)
