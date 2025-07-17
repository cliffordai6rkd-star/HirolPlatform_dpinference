from setuptools import find_packages, setup

package_name = 'hand_eye_calib'

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
    maintainer='Haotian Liang',
    maintainer_email='ryzeliang@163.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
	entry_points={
    	"console_scripts": [
        	"hand_eye_calib_node = hand_eye_calib.hand_eye_calib_node:main",
    ],
},
)
