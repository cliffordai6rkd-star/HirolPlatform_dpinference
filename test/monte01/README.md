```
conda create -n [your env name] python=3.10
conda install pinocchio eigenpy -c conda-forge 
pip3 install mujoco
```

Then run test script:
sim
```
python test_sim2real.py
```

sim + hardware
```
python test_sim2real.py --use_real_robot
```

For camera usage,
1. Please use docker container from `docker/Monte01/docker-compose.yml`, add docker extensions from vscode, then you can do `Run Service` in the .yml
Once you did it, find the monte01-... container in the containers list, and hit `Attach Shell` in the right mouse button menu.

2. https://docs.ros.org/en/humble/Concepts/Intermediate/About-Domain-ID.html calc udp port range,
say, if we use `export ROS_DOMAIN_ID=13` on both machines, we will get some port range from 10650~10687 on the page above,

then, some specific firewall rules should be added to our system:
```
sudo ufw allow 10650:10687/udp
sudo ufw enable # restart ufw to let it work
ros2 topic list # and now will see the full list~
```