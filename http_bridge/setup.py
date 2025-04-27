from setuptools import find_packages, setup

package_name = 'http_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'flask',
        'numpy',
    ],
    zip_safe=True,
    maintainer='Mario Martinez',
    maintainer_email='mario.mtz@manchester-robotics.com',
    description='HTTP bridge for RC car',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rc_server_node = http_bridge.rc_server_node:main',
            'pose_estimator = http_bridge.pose_estimator:main',
        ],
    },
)
