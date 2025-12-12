from catchers_vision_interfaces.msg import ArucoMarkers
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray, TransformStamped
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster
import tf_transformations


class arucoNode(Node):

    def __init__(self):
        """
        Aruco Node.

        Node for detection of Aruco markers and publishing them on Rviz.

        Publishers
        ----------
        aruco_poses : geometry_msgs/msg/PoseArray
            Publishes the pose of the Aruco Marker.

        aruco_markers : catchers_vision_interfaces/msg/ArucoMarkers
            Publishes all the ids of detected markers and poses.

        camera/camera/color/image_raw_axes : sensor_msgs/msg/Image
            Publishes the raw image along with the axes of the aruco marker drawn.

        Subscribers
        -----------
        camera/camera/color/camera_info : sensor_msgs/msg/Image
            Subscribes to camera info topic of the camera.

        camera/camera/color/image_raw : sensor_msgs/msg/Image
            Subscribes to the raw image from the camera
        """
        super().__init__('aruco_detect')

        # Parameter declaration # noqa: E26
        self.declare_parameter(
            'marker_size',
            0.1,
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)
        )
        self.declare_parameter(
            'aruco_dictionary_id',
            'DICT_5X5_250',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        self.declare_parameter(
            'image_topic',
            '/camera/image_raw',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        self.declare_parameter(
            'camera_info_topic',
            '/camera/camera_info',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        self.declare_parameter(
            'camera_frame',
            '',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        self.declare_parameter(
            'aruco_ids',
            [-1],
            ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER_ARRAY),
        )

        # Load Params # noqa: E26
        self.marker_size = (
            self.get_parameter('marker_size').get_parameter_value().double_value
        )
        self.aruco_ids = (
            self.get_parameter('aruco_ids').get_parameter_value().integer_array_value
        )
        self.camera_frame = (
            self.get_parameter('camera_frame').get_parameter_value().string_value
        )
        self.camera_info_topic = (
            self.get_parameter('camera_info_topic').get_parameter_value().string_value
        )
        self.image_topic = self.get_parameter('image_topic').value
        self.aruco_fam = self.get_parameter('aruco_dictionary_id').value

        # Subscribers # noqa: E26
        self.get_logger().info(f'Aruco ids  is: {self.aruco_ids}')
        self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.info_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(Image, self.image_topic, self.image_callback, 100)

        # Publishers # noqa: E26
        self.poses_pub = self.create_publisher(PoseArray, 'aruco_poses', 10)
        self.markers_pub = self.create_publisher(ArucoMarkers, 'aruco_markers', 10)
        self.image_pub = self.create_publisher(Image, self.image_topic + '_axes', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.info_msg = None
        self.intrinsic_mat = None
        self.distortion = None
        self.bridge = CvBridge()
        self.publish_tf = True

        try:
            dictionary_id = cv2.aruco.__getattribute__(self.aruco_fam)
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        except Exception as e:
            self.get_logger().error(f'Failed to load dictionary {self.aruco_fam}: {e}')
            raise

    def info_callback(self, info_msg):
        """
        Info callback for reading the CameraInfo.

        Args
        ----
        info_msg : sensor_msgs/msg/CameraInfo

        Returns
        -------
        None
        """
        self.info_msg = info_msg
        self.intrinsic_mat = np.reshape(np.array(self.info_msg.k), (3, 3))
        self.distortion = np.array(self.info_msg.d)

    def image_callback(self, img_msg):
        """
        Image callback to get the raw image from camera.

        Args
        ----
        img_msg : sensor_msgs/msg/Image

        Returns
        -------
        None
        """
        if self.info_msg is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')
            return

        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(gray)
        markers = ArucoMarkers()
        pose_array = PoseArray()
        markers.header.frame_id = self.camera_frame
        markers.header.stamp = img_msg.header.stamp
        pose_array.header.frame_id = self.camera_frame
        pose_array.header.stamp = img_msg.header.stamp
        cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
        rvecs = []
        tvecs = []

        # Define object points for the marker (real-world coordinates)
        half = self.marker_size / 2.0
        obj_points = np.array(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
            dtype=np.float32,
        )

        for c in corners:
            img_points = c.reshape((4, 2)).astype(np.float32)

            success, rvec, tvec = cv2.solvePnP(
                obj_points,
                img_points,
                self.intrinsic_mat,
                self.distortion,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if success:
                rvecs.append(rvec)
                tvecs.append(tvec)

        rvecs = np.array(rvecs)
        tvecs = np.array(tvecs)

        if len(rvecs) > 0:
            cv2.drawFrameAxes(
                cv_image,
                self.intrinsic_mat,
                self.distortion,
                rvecs[0],
                tvecs[0],
                self.marker_size * 0.5,
            )

        if ids is None or len(ids) == 0:
            return

        self.get_logger().info(f'Ids are: \n{ids}')
        for marker_id in ids:
            marker_int = int(marker_id[0])
            if marker_int not in self.aruco_ids:
                return

        for i, marker_id in enumerate(ids):
            pose = Pose()
            tvec = tvecs[i].squeeze()
            pose.position.x = float(tvec[0])
            pose.position.y = float(tvec[1])
            pose.position.z = float(tvec[2])

            rvec = rvecs[i].squeeze()
            rot_matrix = np.eye(4)
            rot_matrix[0:3, 0:3] = cv2.Rodrigues(rvec)[0]
            quat = tf_transformations.quaternion_from_matrix(rot_matrix)

            pose.orientation.x = float(quat[0])
            pose.orientation.y = float(quat[1])
            pose.orientation.z = float(quat[2])
            pose.orientation.w = float(quat[3])

            pose_array.poses.append(pose)
            markers.poses.append(pose)

            markers.marker_ids.append(int(marker_id[0]))

            self.get_logger().info(
                f'Detected {len(ids)} markers: {ids.flatten().tolist()}'
            )

            if self.publish_tf:
                self._publish_transform(
                    pose, marker_id[0], markers.header.frame_id, img_msg.header.stamp
                )

        final_image = self.bridge.cv2_to_imgmsg(cv_image)

        self.poses_pub.publish(pose_array)
        self.markers_pub.publish(markers)
        self.image_pub.publish(final_image)

    def _publish_transform(self, pose, marker_id, frame_id, stamp):
        """
        Publish the aruco marker transform to the TF tree.

        Args
        ----
        pose : geometry_msgs/msg/Pose
            Pose of the marker.

        marker_id : int64
            Id of the marker.

        frame_id : string
            Frame of the camera the transform is published in.

        stamp : float
            Timestamp.
        """
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = frame_id
        transform.child_frame_id = f'aruco_marker_{int(marker_id)}'
        transform.transform.translation.x = pose.position.x
        transform.transform.translation.y = pose.position.y
        transform.transform.translation.z = pose.position.z
        transform.transform.rotation.x = pose.orientation.x
        transform.transform.rotation.y = pose.orientation.y
        transform.transform.rotation.z = pose.orientation.z
        transform.transform.rotation.w = pose.orientation.w
        self.tf_broadcaster.sendTransform(transform)


def main():
    """Aruco node Runner."""
    rclpy.init()
    node = arucoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
