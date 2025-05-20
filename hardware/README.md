FR3
使用[libfranka-sim](git@github.com:BigJohnn/libfranka-sim.git) 或者 [panda_py](https://jeanelsner.github.io/panda-py/panda_py.html)进行控制
在控制主机上切换到实时内核，
```
pro attach
pro enable realtime-kernel
```

然后

```
pip install -r requirements.txt
```

UnitreeG1

```
ssh unitree@192.168.123.164 # passwd 123

sudo nmcli radio wifi on # Turn the wifi on/off
sudo nmcli device wifi connect XXXXX password XXXXX # Join WiFi network 
# 如果lan连接主路由器的网络192.168.100.1，
# 那么wlan只能连二级路由器

网络192.168.31.1
sudo timedatectl set-ntp no 
sudo timedatectl set-time "2025-04-29 17:07:59" # 改成实际时间
sudo timedatectl set-ntp yes # Set time via NTP

ping www.baidu.com # 此时应能上网
```