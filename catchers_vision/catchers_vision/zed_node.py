import os

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, TransformStamped
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster

from . import zed


class ZedTracker(Node):

    def __init__(self):
        super().__init__('zed_tracker')

        pkg_share = get_package_share_directory('catchers_vision')
        model_path = os.path.join(pkg_share, 'model', 'yolo.onnx')
        model_path = ''
        self.zed_obj = zed.ZedImage(model_path)

        #Establish broadcaster # noqa: E26
        self.broadcaster = TransformBroadcaster(self)

        #Publishers # noqa: E26
        self.raw_img_pub = self.create_publisher(
            Image,
            'zed/image_raw',
            10
        )
        self.det_img_pub = self.create_publisher(
            Image,
            'zed/detect_image',
            10
        )
        self.ball_pos_pub = self.create_publisher(
            PointStamped,
            'ball_pose',
            10
        )
        
        self.camera_info_pub = self.create_publisher(
            CameraInfo,
            'zed/camera_info',
            10
        )

        #Timer callback # noqa: E26
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.bridge = CvBridge()
        self.calib = self.make_camera_info('zed_camera_link')

    def timer_callback(self):
        """Timer callback."""
        time_now = self.get_clock().now().to_msg()
        self.calib.header.stamp = time_now
        self.camera_info_pub.publish(self.calib)

        raw_img, det_img, pos = self.zed_obj.get_frame()

        if raw_img is None:
            return

        msg_raw = self.bridge.cv2_to_imgmsg(raw_img, encoding='bgr8')
        msg_raw.header.stamp = time_now
        msg_raw.header.frame_id = 'zed_camera_link'
        self.raw_img_pub.publish(msg_raw)

        msg_det = self.bridge.cv2_to_imgmsg(det_img, encoding='bgr8')
        msg_det.header.stamp = time_now
        msg_det.header.frame_id = 'zed_camera_link'
        self.det_img_pub.publish(msg_det)
        if pos:
            self.broadcast_ball(pos, time_now)
        else:
            self.broadcast_ball([-1.0,-1.0,-1.0], time_now)
            

    def broadcast_ball(self, location, time):
        """Broadcast ball tf to tf tree."""
        pt = PointStamped()
        pt.header.stamp = time
        pt.header.frame_id = 'zed_camera_link'
        pt.point.x = float(location[0])
        pt.point.y = float(location[1])
        pt.point.z = float(location[2])
        self.ball_pos_pub.publish(pt)

        transform = TransformStamped()
        transform.header.stamp = time
        transform.header.frame_id = 'zed_camera_link'
        transform.child_frame_id = 'ball'
        transform.transform.translation.x = float(location[0])
        transform.transform.translation.y = float(location[1])
        transform.transform.translation.z = float(location[2])
        transform.transform.rotation.w = 1.0
        self.broadcaster.sendTransform(transform)
    
    def make_camera_info(self, frame_id):
        """Make CameraInfo msg."""
        calib = self.zed_obj.calib
        left_cam = self.zed_obj.left_cam
        width = self.zed_obj.cam_info.camera_configuration.resolution.width
        height = self.zed_obj.cam_info.camera_configuration.resolution.height

        msg = CameraInfo()
        msg.header.frame_id = frame_id
        msg.width = width
        msg.height = height

        
        fx = left_cam.fx
        fy = left_cam.fy
        cx = left_cam.cx
        cy = left_cam.cy
        msg.k = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0,
        ]

        # R = calib.R  
        # msg.r = list(R)  
       
        # Tx = calib.T[0]
        # Tx_pixels = -fx * Tx  
        # msg.p = [
        #     fx, 0.0, cx, Tx_pixels,
        #     0.0, fy, cy, 0.0,
        #     0.0, 0.0, 1.0, 0.0,
        # ]

        msg.distortion_model = "plumb_bob"
        msg.d = list(left_cam.disto) 

        return msg


def main(args=None):
    """Entry point for the zed_tracker node."""
    rclpy.init(args=args)
    node = ZedTracker()
    rclpy.spin(node)
    node.zed_obj.close()
    rclpy.shutdown()


if __name__ == '__main__':
    import sys
    main(sys.argv)
