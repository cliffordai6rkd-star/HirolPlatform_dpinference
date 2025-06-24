

1.run sim
```
python test_sim2real.py -g
```

2.run harware
```
python test_sim2real.py --robot-ip 192.168.1.101
```

3.run both
```
python test_sim2real.py --robot-ip 192.168.1.101 -g
```







P.S. ensure your dm_robotics_panda && panda_py work!! 
```
conda install pinocchio eigenpy -c conda-forge

cd dependencies/dm_robotics_panda
pip install -e .
```
And then maybe you should reinstall your panda_py.