from catchers_vision.trajectory_prediction import LSMADParabola
from geometry_msgs.msg import Point, PoseStamped
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker

#x:1.2242397229384518, y:-1.9332898148673308, z:1.5516697580441008 # noqa : E26


class TrajPred(Node):
    """Inference trajectory of ball."""

    def __init__(self):
        """Initialize the ball tracking node."""
        super().__init__('traj_pred')
        self.get_logger().info('traj_pred')
        self.rls = LSMADParabola(
            [-1.0, 1.0],
            [-1.0, 1.0],
            [0, 0.1],
            N=7,
            N_best=4,
            v_gate=None
        )

        self.plot = self.create_service(
            Empty,
            'plot',
            self.plot_callback
        )
        self._tmr = self.create_timer(0.001, self.timer_callback)
        self.t_i = None
        self.theta = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
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

        self.declare_parameter(
            'actual_marker_topic',
            '/ball_actual'
        )
        self.declare_parameter(
            'pred_marker_topic',
            '/ball_pred'
        )

        self.ball_actual_topic = self.get_parameter('actual_marker_topic').value
        self.ball_pred_topic = self.get_parameter('pred_marker_topic').value

        self.ball_actual_pub = self.create_publisher(
            Marker,
            self.ball_actual_topic,
            10
        )

        self.ball_pred_pub = self.create_publisher(
            Marker,
            self.ball_pred_topic,
            10
        )
        
        self.reset_srv = self.create_service(
            Empty,
            'reset_throw',
            self.reset_callback
        )

        self.goal_pose_pub = self.create_publisher(
            PoseStamped,
            'goal_pose',
            10
        )
        
        self.points = []
        self.pred = []

    def timer_callback(self):
        trans = self.query_frame('base', 'ball')
        if trans is None:
            self.get_logger().warn('Transform base→ball not available')
            return

        x = trans.transform.translation.x
        y = trans.transform.translation.y
        z = trans.transform.translation.z
        t_msg = trans.header.stamp
        t = t_msg.sec + t_msg.nanosec * 1e-9
        loc = np.array([x, y, z])
        if np.all(np.isclose(loc, self.default_val)) or np.all(
            np.isclose(loc, self.prev_loc)
        ):
            return
        self.get_logger().info(f'meas is: x:{x}, y:{y}, z:{z}')
        self.add_point(x, y, z, 'actual')
        self.publish_marker('actual')

        self.theta = self.rls.update(x, y, z, t)
        goal, quat = self.rls.calc_goal([0.0, 0.0, 0.0, 1.0])
        
        if not np.any(np.isnan(goal)):
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'base'
            goal_pose.header.stamp = t_msg
            goal_pose.pose.position.x = goal[0]
            goal_pose.pose.position.y = goal[1]
            goal_pose.pose.position.z = goal[2]
            goal_pose.pose.orientation.x = quat[0]
            goal_pose.pose.orientation.y = quat[1]
            goal_pose.pose.orientation.z = quat[2]
            goal_pose.pose.orientation.w = quat[3]
            self.goal_pose_pub.publish(goal_pose)  
        
        if self.t_i is None:
            self.t_i = t
        self.x_meas.append(x)
        self.y_meas.append(y)
        self.z_meas.append(z)
        self.t.append(t - self.t_i)
        self.prev_loc = loc
        
        if not np.any(np.isnan(self.theta)):
            self.pred.clear()
            t = np.linspace(0, self.t[-1])
            model = self.theta
            x_pred = model[0]*t + model[1]
            y_pred = model[2]*t + model[3]
            z_pred = model[4]*(t**2) + model[5]*t + model[6]

            for x, y, z in zip(x_pred, y_pred, z_pred):
                self.add_point(x, y, z, 'pred')

            self.publish_marker('pred')
            
        

    def plot_callback(self, request, response):
        """Plot callback."""
        goal, quat = self.rls.calc_goal([0.0, 0.0, 0.0, 1.0])
        t = np.linspace(0, self.t[-1])
        model = self.theta
        x_pred = model[0]*t + model[1]
        y_pred = model[2]*t + model[3]
        z_pred = model[4]*(t**2) + model[5]*t + model[6]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_pred, y_pred, z_pred, linewidth=2)
        ax.scatter(self.x_meas, self.y_meas, self.z_meas, alpha=0.5, color='red')
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

    def add_point(self, x, y, z, pub_type):
        """Add points to array to publish Markers."""
        p = Point()
        p.x = x
        p.y = y
        p.z = z
        if pub_type == 'actual':
            self.points.append(p)
        else:
            self.pred.append(p)

    def publish_marker(self, pub_type):
        """Marker publisher."""
        m = Marker()
        m.header.frame_id = 'base'
        m.ns = pub_type
        m.id = 0
        m.type = Marker.SPHERE_LIST

        m.scale.x = 0.04
        m.scale.y = 0.04
        m.scale.z = 0.04

        m.color.a = 1.0
        if pub_type == 'actual':
            m.color.g = 1.0
            m.points = self.points
            self.ball_actual_pub.publish(m)

        else:
            m.color.b = 1.0
            m.points = self.pred
            self.ball_pred_pub.publish(m)
    
    def reset_callback(self, request, response):
        self.get_logger().info('Resetting trajectory predictor for new throw')

        self.rls.reset()

        self.t_i = None
        self.theta = np.array([np.nan, np.nan, np.nan,
                            np.nan, np.nan, np.nan, np.nan])

        self.x_meas.clear()
        self.y_meas.clear()
        self.z_meas.clear()
        self.t.clear()

        self.points.clear()
        self.pred.clear()

        self.prev_loc = self.default_val

        self.publish_marker('actual')
        self.publish_marker('pred')

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
