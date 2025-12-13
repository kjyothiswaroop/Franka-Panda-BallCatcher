from geometry_msgs.msg import PoseStamped, TransformStamped
from launch import LaunchDescription
from launch_ros.actions import Node
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from catchers_vision.traj_pred_node import TrajPred

import launch_testing
import pytest
import rclpy
import time
import unittest


@pytest.mark.rostest
def generate_test_description():
    return (
        LaunchDescription([
            Node(package='catchers_vision',
                 executable='traj_pred_node'),
            launch_testing.actions.ReadyToTest()
            ])
    )


class TestTrajPre(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_node')

        self.static_broadcaster = StaticTransformBroadcaster(self.node)
        world_base_tf = TransformStamped()
        world_base_tf.header.stamp = self.node.get_clock().now().to_msg()
        world_base_tf.header.frame_id = 'world'
        world_base_tf.child_frame_id = 'base'
        world_base_tf.transform.translation.z = 1.0
        self.static_broadcaster.sendTransform(world_base_tf)

        self.d = 0.0
        self.broadcaster = TransformBroadcaster(self.node)

    def tearDown(self):
        self.node.destroy_node()

    def timer_callback(self):
        self.d += 0.001
        base_ball_tf = TransformStamped()
        base_ball_tf.header.frame_id = 'base'
        base_ball_tf.child_frame_id = 'ball'
        base_ball_tf.transform.translation.x = 1.0 + self.d
        base_ball_tf.transform.translation.y = 0.0
        base_ball_tf.transform.translation.z = 1.0 + self.d

        time = self.node.get_clock().now().to_msg()
        base_ball_tf.header.stamp = time

        self.broadcaster.sendTransform(base_ball_tf)

    def test_trajectory_prediction(self):
        """Check whether goal pose messages published."""
        goal_pose_buffer = []
        sub = self.node.create_subscription(
            PoseStamped,
            'goal_pose',
            lambda msg: goal_pose_buffer.append(msg),
            10
        )
        timer = self.node.create_timer(
            0.01,
            self.timer_callback
        )
        try:
            end_time = time.time() + 10
            while time.time() < end_time:
                rclpy.spin_once(self.node, timeout_sec=1)
                # Comment this if statement to see what happens in Rviz
                if len(goal_pose_buffer) > 30:
                    break
            self.assertGreater(len(goal_pose_buffer), 30)
        finally:
            self.node.destroy_subscription(sub)
            self.node.destroy_timer(timer)