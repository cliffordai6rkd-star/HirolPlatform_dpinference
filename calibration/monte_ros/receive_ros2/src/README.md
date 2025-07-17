# prep part
```
bash
sudo apt install ros-humble-gazebo-ros
sudo apt install ros-humble-gazebo-ros-pkgs
sudo apt install ros-humble-moveit-visual-tools
sudo apt-get install ros-humble-gazebo-ros2-control
sudo apt-get install ros-humble-tf-transformations


```
## hand eye calibration
    get relationship of camera and base

## urdf
1. set a virtual tool as tcp, distance is 0.22 or 0.232, no need any physical property
   we can use tool or gripper_base as parent link

2. use hand eye calibration matrix, set position and pose of realsensed435
   parent link 'base'(eye to hand) or tool0(eye in hand)
   
   note:
    calibration by mine method will get  realsense 'camera_color_optical_frame' to base
   or not camera_link to base. So we ought to switch ['camera_color_optical_frame' to base] to
   camera link to base.
   ```bash
   ros2 run tf2_ros tf2_echo  camera_color_optical_frame camera_link
   ```
   get $$ ^{camera\_color\_optical\_frame}T_{camera\_link} $$, 
   then $$ ^{base}T_{camera\_color\_optical\_frame} $$ @ $$ ^{camera\_color\_optical\_frame}T_{camera\_link} $$
    get $$ ^{base}T_{camera\_link} $$ 
    turn [camera link to base] to x y z rpy


3. set up collision matirx
   
## scene set
1.

2.

3.




 

# flow of grasping 

## start robot arm rviz
```bash
ros2 launch bringup simulation.launch.py
```
## save img
```bash
ros2 run vision save_img
```
## Real-time Grasp Pose Inference and Visualization (Optional)
``` bash
ros2 launch gnb gnb_server.launch.py
```
## grasp detect
```bash
cd graspnet
command bash_demo
```
## pub msg:result of graspnet gg
```bash
ros2 launch vision tf_pub.launch.py
```


### （Sim）Real word dont need.
## since we use gazebo_ros_link_attacher, we should change src/ur5e_gripper_control/config/obj_class.yaml

    # params.yaml
obj_grasp:
  ros__parameters:
    obj_class: "salt"   #change it banana or any other

## see tf, check coordinate of target obj

## execuate motion and grasp in gazebo
```bash
ros2 launch ur5e_gripper_control demo1.launch.py
```

## real should change obj_graso.cpp. we dont need attach deattach

