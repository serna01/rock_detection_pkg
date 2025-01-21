import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Pose2D, Detection2DArray, Detection2D, ObjectHypothesis, BoundingBox2D
from vision_msgs.msg import VisionInfo
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import os
from typing import List, Dict
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
import time
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker

## /leo08/camera/image_rect_color'
##/leo08/zed2/zed_node/left/image_rect_color

class ROS2RockDetectorNode(Node):
    def __init__(self, topic_name: str = '/leo08/zed2/zed_node/left/image_rect_color'):
        super().__init__('ros2_rock_detector_node')
        self.get_logger().info('ROS2RockDetectorNode initialized')

        self.subscription = self.create_subscription(
            Image, topic_name, self.image_callback, 10)
        self.get_logger().info(f'Subscribed to topic: {topic_name}')

        self.publisher = self.create_publisher(Detection2DArray, 'rock_detections', 10)
        self.get_logger().info('Created publisher for rock_detections topic')

        self.marker_publisher = self.create_publisher(MarkerArray, '/rock_marker_array', 10)
        self.get_logger().info('Created publisher for marker array')

        self.vision_info_publisher = self.create_publisher(VisionInfo, 'rock_detector_info', 10)
        self.get_logger().info('Created publisher for rock_detector_info topic')

        self.last_published_msg = None

        # Load model
        model_weights_path = os.path.expanduser('~/Rockdetect/runs/detect/train9/weights/best.pt')
        #self.get_logger().info(f'Loading YOLOv8 model from: {model_weights_path}')
        self.model = YOLO(model_weights_path)
        self.database_version = 1
        self.get_logger().info(f'Model loaded successfully: {model_weights_path}')

    def image_callback(self, msg: Image):
        self.get_logger().info(f'Received Image message at {time.time()}')

        # Convert the ROS Image message to a NumPy array
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        self.get_logger().debug(f'Image shape after reshaping: {img.shape}')

        # Handle 4-channel images (e.g., bgra8)
        if img.shape[2] == 4:
            self.get_logger().info('Image has 4 channels; converting to 3 channels (BGR)')
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.shape[2] == 3:
            self.get_logger().info('Image has 3 channels (BGR), no conversion needed')
        else:
            self.get_logger().error(f'Unexpected number of channels: {img.shape[2]}')
            return  # Exit if the format is not as expected

        # Resize image to speed up detection
        img_resized = cv2.resize(img, (640, 480))
        self.get_logger().debug('Resized image for faster detection')

        # Run YOLOv8 inference on the resized image
        self.get_logger().info('Running YOLOv8-rock inference on the image')
        results_list = self.model(img_resized)

        # Prepare the Detection2DArray message
        self.get_logger().info('Preparing Detection2DArray message')
        detection_msg = Detection2DArray()
        detection_msg.header.stamp = self.get_clock().now().to_msg()
        detection_msg.header.frame_id = msg.header.frame_id  # Use frame_id from the input image

        detection_id = 1  # Counter for unique IDs

        # Create and publish MarkerArray after detecting rocks
        marker_array = MarkerArray()

        for result in results_list:
            boxes_list = self.parse_boxes(result)
            for box in boxes_list:
                detection = Detection2D()
                # Add header information
                detection.header.stamp = self.get_clock().now().to_msg()
                detection.header.frame_id = msg.header.frame_id
                
                # Assign a unique ID
                detection.id = f"rock{detection_id}"
                detection_id += 1

                detection.bbox = box
                detection_msg.detections.append(detection)

                # Create marker for each detection
                try:
                    marker = self.create_marker(detection, detection_id - 1, msg.width, msg.height)
                    marker_array.markers.append(marker)
                except Exception as e:
                    self.get_logger().error(f'Error creating marker: {e}')

        # Publish the marker array only if there are markers
        if marker_array.markers:
            self.marker_publisher.publish(marker_array)
        
        # Publish the detection message
        self.get_logger().info('Publishing Detection2DArray message to rock_detections topic')
        self.publisher.publish(detection_msg)

        # Publish vision info
        self.get_logger().info('Publishing VisionInfo message to rock_detector_info topic')
        vision_info = VisionInfo()
        vision_info.database_version = self.database_version
        self.vision_info_publisher.publish(vision_info)

    def parse_boxes(self, result: Results, confidence_threshold: float = 0.75) -> List[BoundingBox2D]:
        self.get_logger().info('Parsing bounding boxes from Rockdetector model results')
        boxes_list = []

        if result.boxes:
            self.get_logger().debug(f'Extracting {len(result.boxes)} bounding boxes from result.boxes')
            for box_data in result.boxes:
                conf = box_data.conf.item()  # Extract confidence
                if conf >= confidence_threshold:
                    msg = BoundingBox2D()
                    box = box_data.xywh[0]  # Extract x, y, width, height
                    msg.center.position.x = float(box[0])
                    msg.center.position.y = float(box[1])
                    msg.size_x = float(box[2])
                    msg.size_y = float(box[3])
                    boxes_list.append(msg)

        elif result.obb:
            self.get_logger().debug(f'Extracting {result.obb.cls.shape[0]} oriented bounding boxes from result.obb')
            for i in range(result.obb.cls.shape[0]):
                conf = result.obb.conf[i].item()  # Extract confidence
                if conf >= confidence_threshold:
                    msg = BoundingBox2D()
                    box = result.obb.xywhr[i]
                    msg.center.position.x = float(box[0])
                    msg.center.position.y = float(box[1])
                    msg.center.theta = float(box[4])
                    msg.size_x = float(box[2])
                    msg.size_y = float(box[3])
                    boxes_list.append(msg)

        self.get_logger().debug(f'Parsed {len(boxes_list)} bounding boxes above confidence threshold {confidence_threshold}')
        return boxes_list

    def create_marker(self, detection, marker_id, original_width, original_height):
        marker = Marker()
        marker.header.frame_id = detection.header.frame_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.id = marker_id
        
        # Normalize coordinates to meters or use camera's intrinsic parameters
        marker.pose.position.x = detection.bbox.center.position.x / original_width
        marker.pose.position.y = detection.bbox.center.position.y / original_height
        marker.pose.position.z = 1.0  # Set a reasonable distance in front of camera
        
        marker.scale.x = detection.bbox.size_x / original_width
        marker.scale.y = detection.bbox.size_y / original_height
        marker.scale.z = 0.1
        
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.5
        
        marker.lifetime.sec = 5  # Optional: set marker lifetime
        
        return marker

def main(args=None):
    rclpy.init(args=args)
    rock_detector_node = ROS2RockDetectorNode()
    rclpy.spin(rock_detector_node)
    rock_detector_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()