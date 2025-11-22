"""Setup file."""
from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'catchers_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='KThompson2002',
    maintainer_email='kylezthompson@gmail.com',
    description='Vision package for ball tracking in Catchers',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ball_track = catchers_vision.ball_track:main',
            'traj_pred_node = catchers_vision.traj_pred_node:main'
        ],
    },
)
