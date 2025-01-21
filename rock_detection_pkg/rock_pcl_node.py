import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2
import psutil
from geometry_msgs.msg import PointStamped, PoseStamped, TransformStamped
import tf2_ros
import tf2_geometry_msgs
from builtin_interfaces.msg import Duration

class RockDepthSubscriber(Node):
    def __init__(self):
        super().__init__('rock_depth_subscriber')
        
        # Publish the static transform
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.publish_static_transform()
        # Initialize TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info("TF2 buffer and listener initialized.")
        
        # Subscribe to the rock depth topic
        self.subscription = self.create_subscription(
            String,
            '/rock_depth_topic',  # Replace with the actual topic
            self.rock_depth_callback,
            10)

        # Subscribe to the CameraInfo topic to get camera intrinsics
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/leo08/zed2/zed_node/left/camera_info',  # ZED Camera Info topic
            self.camera_info_callback,
            10)
        
        # Subscribe to the pose topic
        self.pose_subscription = self.create_subscription(
            PoseStamped,
            '/vrpn_mocap/alejandro_test/rectified/pose',
            self.pose_callback,
            10)

        self.point_cloud_publisher = self.create_publisher(PointCloud2, '/rock_point_cloud', 10)
        self.world_point_cloud_publisher = self.create_publisher(PointCloud2, '/rock_world', 10)
        

        # Default camera parameters (will be updated after receiving CameraInfo)
        self.fx = 262.2794
        self.fy = 262.2794
        self.cx = 319.6014
        self.cy = 174.6729

        # Flag to track if camera info has been logged
        self.camera_info_initialized = False

        # Keep track of detected points and threshold for new points
        self.detected_points = []
        #self.threshold_distance = 0.5  # meters, adjust as needed

        # Set an interval for memory logging (in seconds)
        #self.memory_log_interval = 5  # log every 5 seconds
        

    def camera_info_callback(self, msg):
        # Extract camera intrinsics (focal lengths and principal point)
        self.fx = msg.k[0]  # Focal length in x (from k matrix)
        self.fy = msg.k[4]  # Focal length in y (from k matrix)
        self.cx = msg.k[2]  # Principal point x (from k matrix)
        self.cy = msg.k[5]  # Principal point y (from k matrix)

        # Log camera intrinsics only once
        if not self.camera_info_initialized:
            self.get_logger().info(f"Camera intrinsics initialized: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
            self.camera_info_initialized = True  # Set flag to True after logging

    def rock_depth_callback(self, msg):
        # Example message format: "x, y, depth"
        x, y, depth = self.parse_rock_depth(msg.data)
        if x is None or y is None or depth is None:
            return

        point_3d = self.convert_to_3d(x, y, depth)

        # Check if the point is new or close to an existing point
        #if self.is_new_point(point_3d):
        # Add the new point to the list of detected points
        self.detected_points.append(point_3d)
        self.get_logger().info(f"New point detected: {point_3d}")

        # Create a PointCloud2 message from the 3D points
        cloud_msg = self.create_point_cloud(self.detected_points)
        self.point_cloud_publisher.publish(cloud_msg)
        self.get_logger().info(f"Publishing point cloud with {len(self.detected_points)} points")
        
        
        # Convert points to world frame and publish
        if self.tf_buffer.can_transform("vrpn_mocap/alejandro_test", "leo08/zed2_camera_link", rclpy.time.Time()):
            self.get_logger().info("World Transform is available.")
            world_points = self.convert_points_to_world(self.detected_points)
            world_cloud_msg = self.create_point_cloud(world_points)
            self.world_point_cloud_publisher.publish(world_cloud_msg)
            self.get_logger().info(f"Publishing world point cloud with {len(world_points)} points")
        else:
            self.get_logger().warn("World Transform is not available. Skipping world frame conversion.")

    def parse_rock_depth(self, msg):
        # Parse the message to extract the pixel coordinates and depth
        try:
            parts = msg.split(',')
            x, y, depth = float(parts[0]), float(parts[1]), float(parts[2])
            return x, y, depth
        except Exception as e:
            self.get_logger().warn(f"Failed to parse message: {e}")
            return None, None, None

    def convert_to_3d(self, x, y, depth):
        # Convert pixel coordinates (x, y) and depth to 3D world coordinates
        if self.fx == 0.0 or self.fy == 0.0:
            self.get_logger().warn("Camera intrinsics are not initialized yet!")
            return np.array([0.0, 0.0, 0.0])

        # Compute the 3D point using the camera intrinsics and
        # Convert from OpenCV coordinates to ROS camera coordinates
        # OpenCV (x, y) -> ROS (-y, -z), depth -> x
        self.x = depth
        self.y = -(x - self.cx) * depth / self.fx
        self.z = -(y - self.cy) * depth / self.fy

        return np.array([self.x, self.y, self.z])
    
    def create_point_cloud(self, points):
        # Ensure points is a list of 3D points (x, y, z)
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'leo08/zed2_left_camera_frame'  # Replace with the correct frame id

        # Convert points to a list of tuples (x, y, z)
        pc_data = [(point[0], point[1], point[2]) for point in points]

        # Create and return the PointCloud2 message
        cloud_msg = point_cloud2.create_cloud_xyz32(header, pc_data)
        return cloud_msg

    def is_new_point(self, point_3d):
        """
        Check if the incoming 3D point is sufficiently far from existing points
        using a Euclidean distance threshold.
        """
        for detected_point in self.detected_points:
            distance = np.linalg.norm(detected_point - point_3d)
            if distance < self.threshold_distance:
                return False  # Point is too close to an existing one
        return True  # New point, sufficiently far

    def log_memory_usage(self):
        """Log the current memory usage."""
        memory_info = psutil.virtual_memory()
        self.get_logger().info(f"Memory usage: {memory_info.percent}%")
        
    def publish_static_transform(self):
        static_transform = TransformStamped()
        static_transform.header.stamp = self.get_clock().now().to_msg()
        static_transform.header.frame_id = 'vrpn_mocap/alejandro_test'
        static_transform.child_frame_id = 'leo08/zed2_camera_link'
        static_transform.transform.translation.x = -0.004
        static_transform.transform.translation.y = -0.060
        static_transform.transform.translation.z = 0.037
        static_transform.transform.rotation.x = 0.0
        static_transform.transform.rotation.y = -0.017
        static_transform.transform.rotation.z = 0.0
        static_transform.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(static_transform)
        self.get_logger().info("Published static transform")
        
    def pose_callback(self, msg):
        # Handle the pose message
        self.get_logger().debug(f"Received pose: {msg}")
        
    def convert_points_to_world(self, points):
        # Convert points from camera frame to world frame
        world_points = []
        for point in points:
            # Apply the transform using the TF2-based apply_transform function
            transformed_point = self.apply_transform(point)
            if transformed_point is not None:
                world_points.append(transformed_point)
                self.get_logger().info(f"Transformed point: {transformed_point}")
        return world_points
    
    def apply_transform(self, point):
        # Create a PointStamped message for the input point
        point_stamped = PointStamped()
        point_stamped.header.frame_id = "leo08/zed2_camera_link"
        point_stamped.header.stamp = self.get_clock().now().to_msg()
        point_stamped.point.x = point[0]
        point_stamped.point.y = point[1]
        point_stamped.point.z = point[2]

        try:
            # Transform the point to the world frame
            timeout = rclpy.duration.Duration(seconds=1.0)
            transformed_point = self.tf_buffer.transform(
                point_stamped, "vrpn_mocap/alejandro_test", timeout=timeout
            )
            self.get_logger().info(f"Transformed point: {transformed_point.point}")
            # Extract the transformed coordinates
            return np.array([transformed_point.point.x, transformed_point.point.y, transformed_point.point.z])
        except tf2_ros.TransformException as ex:
            self.get_logger().error(f"Transform error: {ex}")
        return None

def main(args=None):
    rclpy.init(args=args)

    rock_depth_subscriber = RockDepthSubscriber()

    rclpy.spin(rock_depth_subscriber)

    rock_depth_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
