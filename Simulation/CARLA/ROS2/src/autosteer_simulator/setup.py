from setuptools import setup

package_name = 'autosteer_simulator'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='JITERN',
    maintainer_email='limjitern@gmail.com',
    description='Autosteer simulator for CARLA simulation in ROS',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'autosteer_simulator_node = autosteer_simulator.autosteer_simulator_node:main',
        ],
    },
)
