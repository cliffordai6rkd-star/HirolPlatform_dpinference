KDL: 

```
ln -s /usr/lib/python3/dist-packages/PyKDL.cpython-38-x86_64-linux-gnu.so /path/to/vk/lib/python3.8/site-packages/PyKDL.cpython-38-x86_64-linux-gnu.so
export PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPAT
```

```
cd motion
g++ -Wall -shared -I/usr/include/eigen3/ -fPIC `python3 -m pybind11 --includes` -Itime_optimal_trajectory/ -o trajectory_planner`python3-config --extension-suffix` trajectory_planner.cpp time_optimal_trajectory/Path.cpp time_optimal_trajectory/Trajectory.cpp
```