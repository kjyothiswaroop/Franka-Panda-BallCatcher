from catchers_vision.trajectory_prediction import RLSParabola
from geometry_msgs.msg import PointStamped
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class TrajPred(Node):
    """Inference trajectory of ball."""

    def __init__(self):
        """Initialize the ball tracking node."""
        super().__init__('traj_pred')
        self.get_logger().info('traj_pred')
        self.ball_pose_sub = self.create_subscription(PointStamped,
                                                      'ball_pose',
                                                      self.ball_track_callback,
                                                      10)
        self.rls = RLSParabola([-0.25, 0.25], [-0.25, 0.25], [0, 0.1])
        self.plot = self.create_service(Empty, 'plot', self.plot_callback)
        self.theta = None
        self.x_meas = []
        self.y_meas = []
        self.z_meas = []
        self.t = []
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)

    def ball_track_callback(self, pt):
        """Ball tracking subscriber."""
        t_msg = pt.header.stamp
        t = t_msg.sec + t_msg.nanosec * 1e-9
        z_cam_ball = pt.point.z

        if z_cam_ball != -1:
            trans = self.buffer.lookup_transform('ball', 'base', rclpy.time.Time())
            self.get_logger().info(f'Transform is: {trans}')

            if trans is None:
                self.get_logger().warn('Transform base→camera not available')
                return
            else:
                x = trans.transform.translation.x
                y = trans.transform.translation.y
                z = trans.transform.translation.z

                self.theta = self.rls.update(x, y, z, t)
                self.x_meas.append(x)
                self.y_meas.append(y)
                self.z_meas.append(z)
                self.t.append(t - self.rls.t_i)

    def plot_callback(self, request, response):
        """Plot callback."""
        t = np.linspace(0, 2)
        model = self.theta
        x_pred = model[0]*t + model[1]
        y_pred = model[2]*t + model[3]
        z_pred = model[4]*(t**2) + model[5]*t + model[6]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_pred, y_pred, z_pred, linewidth=2)
        ax.scatter(self.x_meas, self.y_meas, self.z_meas)
        plt.axis('equal')
        plt.legend(['true', 'pred'])
        plt.show()
        return response


def main(args=None):
    """Entrypoint for pick_node."""
    rclpy.init(args=args)
    node = TrajPred()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    import sys
    main(sys.argv)
