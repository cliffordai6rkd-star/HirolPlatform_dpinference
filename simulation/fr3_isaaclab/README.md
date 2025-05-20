
```
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab/docker

python container.py start ros2 --files x11.yaml --env-files .env.ros2
```
使用vscode 的 container插件，找到isaac-lab-ros2这个container，右键Attach Shell
在新打开的terminal中，执行
```
isaaclab -s
```

