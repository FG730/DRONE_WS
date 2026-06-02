from setuptools import find_packages, setup

package_name = 'vision_tracking'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wsy',
    maintainer_email='wsy@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    scripts=[
        'scripts/move_target_realistic.py',
    ],
    entry_points={
        'console_scripts': [
            'image_viewer = vision_tracking.image_viewer:main',
            'color_tracker = vision_tracking.color_tracker:main',
            'attitude_pn_bearing_servo = vision_tracking.attitude_pn_bearing_servo:main',
            'px4_visual_offboard = vision_tracking.px4_visual_offboard:main',
            'experiment_logger = vision_tracking.experiment_logger:main',
            'plot_experiment = vision_tracking.plot_experiment:main',
            'score_experiment = vision_tracking.score_experiment:main',
        ],
    },
)
