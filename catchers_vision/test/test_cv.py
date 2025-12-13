from geometry_msgs.msg import PointStamped
from launch import LaunchDescription
from launch_ros.actions import Node

import launch_testing
import pytest
import rclpy
import unittest


@pytest.mark.rostest
def generate_test_description():
    return (
        LaunchDescription([
            Node(package='catchers_vision',
                 executable='ball_track'),
            launch_testing.actions.ReadyToTest()
            ])
    )


class TestCV(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_node_2')

    def tearDown(self):
        self.node.destroy_node()

    def test_cv(self):
        """Check CV."""
        self.assertTrue(True)
