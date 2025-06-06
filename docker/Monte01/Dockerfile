FROM ghcr.io/bigjohnn/quest3-monte03-teleop:v1


RUN echo "conda activate tv" >> /home/teleophost/.bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> /home/teleophost/.bashrc
WORKDIR /workspace

RUN http_proxy="" https_proxy="" /home/teleophost/miniconda3/envs/tv/bin/pip install numpy==1.26.4 glog \
mujoco \
xarm-python-sdk