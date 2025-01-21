import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String
import numpy as np

class MinimalDepthSubscriber(Node):
    def __init__(self):
        super().__init__('zed_depth_tutorial')

        # Create depth map subscriber
        self.depth_sub = self.create_subscription(
            Image,
            'leo08/zed2/zed_node/depth/depth_registered',
            self.depth_callback,
            10
        )

        # Create rock detections subscriber
        self.rock_detections_sub = self.create_subscription(
            Detection2DArray,
            '/rock_detections',
            self.rock_detections_callback,
            10
        )

        # Create publisher for rock depth topic
        self.rock_depth_pub = self.create_publisher(String, '/rock_depth_topic', 10)

        # Store latest depth image
        self.latest_depth_image = None

    def depth_callback(self, msg):
        """Callback to store the latest depth image."""
        self.latest_depth_image = msg

    def rock_detections_callback(self, detections_msg):
        """Callback to process bounding boxes and output depths."""
        if self.latest_depth_image is None:
            self.get_logger().warn("No depth image available")
            return

        # Convert the depth image to a numpy array
        depth_image = self.convert_image_to_array(self.latest_depth_image)

        for detection in detections_msg.detections:
            # Get center of the bounding box
            u = int(detection.bbox.center.position.x)
            v = int(detection.bbox.center.position.y)

            # Validate pixel coordinates
            if u < 0 or u >= self.latest_depth_image.width or \
               v < 0 or v >= self.latest_depth_image.height:
                self.get_logger().warn(f"Invalid pixel coordinates: ({u}, {v})")
                continue

            # Calculate linear index
            depth_value = depth_image[v, u]

            # Output the depth value
            self.get_logger().info(
                f"Rock detected at ({u}, {v}) with depth: {depth_value:.2f} m"
            )

            # Publish depth data in x, y, depth format
            depth_msg = String()
            depth_msg.data = f"{u},{v},{depth_value:.2f}"
            self.rock_depth_pub.publish(depth_msg)
            self.get_logger().info(
                f"Rock published in topic /rock_depth_topic: {depth_msg.data}"
            )

    def convert_image_to_array(self, image_msg):
        """Convert an Image message to a numpy array."""
        # Convert byte data to numpy array
        depth_data = np.frombuffer(image_msg.data, dtype=np.float32)
        return depth_data.reshape((image_msg.height, image_msg.width))


def main(args=None):
    rclpy.init(args=args)
    node = MinimalDepthSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
