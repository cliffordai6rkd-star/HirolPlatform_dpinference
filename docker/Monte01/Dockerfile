FROM ghcr.io/bigjohnn/quest3-monte03-teleop:v1


RUN echo "conda activate tv" >> /home/teleophost/.bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> /home/teleophost/.bashrc
WORKDIR /workspace

RUN http_proxy="" https_proxy="" /home/teleophost/miniconda3/envs/tv/bin/pip install numpy==1.26.4 glog \
mujoco \
xarm-python-sdk

USER root
RUN usermod -aG input,plugdev,dialout teleophost
RUN http_proxy="" https_proxy="" apt-get update && http_proxy="" https_proxy="" apt-get install -y libhidapi-dev
# RUN echo 'KERNEL=="hidraw*", SUBSYSTEM=="hidraw", MODE="0664", GROUP="plugdev"' > /etc/udev/rules.d/99-hidraw-permissions.rules
RUN echo 'SUBSYSTEM=="input", GROUP="input", MODE="0660"' > /etc/udev/rules.d/99-hidraw-permissions.rules
RUN echo 'KERNEL=="hidraw*", ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c635", MODE="0666"' >> /etc/udev/rules.d/99-hidraw-permissions.rules

USER teleophost

RUN http_proxy="" https_proxy="" /home/teleophost/miniconda3/envs/tv/bin/pip install pyspacemouse