from catchers_vision.trajectory_prediction import LSMADParabola
from geometry_msgs.msg import PointStamped
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException

#x:1.2242397229384518, y:-1.9332898148673308, z:1.5516697580441008

class TrajPred(Node):
    """Inference trajectory of ball."""

    def __init__(self):
        """Initialize the ball tracking node."""
        super().__init__('traj_pred')
        self.get_logger().info('traj_pred')
        self.rls = LSMADParabola([-0.25, 0.25], [-0.25, 0.25], [0, 0.1],N=6,N_best=3, v_gate=False)
        self.plot = self.create_service(Empty, 'plot', self.plot_callback)
        self._tmr = self.create_timer(0.001, self.timer_callback)
        self.t_i = None
        self.theta = None
        self.x_meas = []
        self.y_meas = []
        self.z_meas = []
        self.t = []
        self.tf_buffer = Buffer()
        self.listener = TransformListener(self.tf_buffer, self)
        self.default_val = np.array([1.2242397229384518,
                                     -1.9332898148673308,
                                     1.5516697580441008])
        self.prev_loc = self.default_val
    
    def timer_callback(self):
        trans = self.query_frame('base', 'ball')
        if trans is None:
            self.get_logger().warn('Transform base→camera not available')
            return
        x = trans.transform.translation.x
        y = trans.transform.translation.y
        z = trans.transform.translation.z
        t_msg = trans.header.stamp
        t = t_msg.sec + t_msg.nanosec * 1e-9
        loc = np.array([x,y,z])
        if np.all(np.isclose(loc, self.default_val)) \
        or np.all(np.isclose(loc, self.prev_loc)):
            return
        self.get_logger().info(f'meas is: x:{x}, y:{y}, z:{z}')
        self.theta = self.rls.update(x, y, z, t)
        if self.t_i is None:
            self.t_i = t
        self.x_meas.append(x)
        self.y_meas.append(y)
        self.z_meas.append(z)
        self.t.append(t - self.t_i)
        self.prev_loc = loc
        
    def plot_callback(self, request, response):
        """Plot callback."""
        t = np.linspace(0, self.t[-1])
        model = self.theta
        x_pred = model[0]*t + model[1]
        y_pred = model[2]*t + model[3]
        z_pred = model[4]*(t**2) + model[5]*t + model[6]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_pred, y_pred, z_pred, linewidth=2)
        ax.scatter(self.x_meas, self.y_meas, self.z_meas,alpha=0.5,color='red')
        plt.axis('equal')
        plt.legend(['true', 'pred'])
        plt.show()
        return response
    
    def query_frame(self, frame_id, child_id):
        """
        Query transform between two frames.

        Parameters
        ----------
        frame_id : str
            Target frame ID.
        child_id : str
            Child frame ID.

        Returns
        -------
        geometry_msgs/msg/TransformStamped | None
            Transform if available,
            else None.

        """
        try:
            t = self.tf_buffer.lookup_transform(
                frame_id,
                child_id,
                rclpy.time.Time())
            return t
        except TransformException as ex:
            ex
            return None


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
