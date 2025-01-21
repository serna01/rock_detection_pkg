from setuptools import setup, find_packages

setup(
    name='rock_detection_pkg',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'setuptools',
        'rclpy',
        'sensor_msgs',
        'vision_msgs',
        'ultralytics',
    ],
    package_dir={'': '.'},
    data_files=[
    ('share/ament_index/resource_index/packages', ['resource/rock_detection_pkg']),
    ('share/rock_detection_pkg', ['package.xml']),
],
    zip_safe=True,
    maintainer='Alejandro Serna Medina',
    maintainer_email='alejandrosernam93@gmail.com',
    description='Package for rock detection in ROS 2',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'rock_detector_node = rock_detection_pkg.rock_detector_node:main',
            'rock_depth_node = rock_detection_pkg.rock_depth_node:main',
            'rock_pcl_node = rock_detection_pkg.rock_pcl_node:main'
        ],
    },
)