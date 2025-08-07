```
sudo apt install portaudio19-dev
```

Quest3连接问题

Q 网络受限，进不去桌面
需要连接VPN，iOS为例，
1.连接wifi，打开热点，
2.使用Shadowrocket，设置，代理共享，记录ip/端口
3.戴上Quest，连接热点，使用2中代理ip/端口，加密使用WPA2
4.会显示8位验证码，使用激活设备的邮箱验证。 #发给jd客服(京选智能数码买手店， 订单号310225994749)，有技术人员帮忙处理 # 对方拒绝提供激活帐号，沟通较困难。

Meta Horizon validate code 5: (Glasses)Settings=>about
Unset proxys maybe
Adb operations like unitree:
These are some of my notes on ADB reverse port forwarding for Quest. I hope they can be of some help to you.
1. Download ADB tools and extract them.
(base) unitree@unitree:~/Downloads/platform-tools-latest-linux/platform-tools$ ls
adb  etc1tool  fastboot  hprof-conv  lib64  make_f2fs  make_f2fs_casefold  mke2fs  mke2fs.conf  NOTICE.txt  source.properties  sqlite3
2. Connect the XR device to the PC via USB and execute the following command:

```
sudo ./adb devices
[sudo] password for unitree: 
- daemon not running; starting now at tcp:5037
- daemon started successfully
List of devices attached
2G0YC1ZF9J0D9D        unauthorized
```

If unauthorized appears, it means the XR device has not been authorized. Put on the XR device and click "Allow" in the pop-up window asking for USB debugging permission. Then execute the command again:

```
$ sudo ./adb devices
List of devices attached
2G0YC1ZF9J0D9D        device
```

Start ADB port reverse forwarding by executing:

```
sudo ./adb -s 2G0YC5ZGC900XR reverse tcp:8012 tcp:8012
8012
```

You can verify the result using:
```
$ sudo ./adb -s 2G0YC1ZF9J0D9D reverse --list
UsbFfs tcp:8012 tcp:8012
```
Configure local HTTPS using a self-signed certificate, this step fuction same as 2.2.3 create certificate
```
$ openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout key.pem -out cert.pem
```
If you want to use Wireless Network Reverse Port Forwarding:
```
sudo ./adb shell ifconfig wlan0
wlan0     Link encap:UNSPEC    Driver cnss_pci
          inet addr:your_xr_device_ip  Bcast:192.168.123.255  Mask:255.255.255.0
          inet6 addr: ***** Scope: Link
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
          RX packets:136560 errors:0 dropped:0 overruns:0 frame:0
          TX packets:115802 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:3000
          RX bytes:156741122 TX bytes:77421476
sudo ./adb tcpip 5566
```
restarting in TCP mode port: 5566
```
sudo ./adb connect 192.168.31.100:5566
```
connected to your_xr_device_ip:5566
```
$ sudo ./adb devices
List of devices attached
your_xr_device_ip:5566        device
sudo ./adb -s 192.168.31.100:5566 reverse tcp:8012 tcp:8012
8012
$ sudo ./adb -s your_xr_device_ip:5566 reverse --list
host-28 tcp:8012 tcp:8012
```

P.S. Quest3 重新激活

```
注：VR需要戴着操作，不然即使显示usb mode也会秒自动开机
VR关机，确定完全黑屏后为关机
然后同时按住    音量（减号）  和   电源键    一直按着开机
直至出现   USB update mode ，然后先单独松开电源键，不要松（减号）音量键，直至减号下滑到其它选项再松开
单按加号/减号移动到  factory reset 项
然后单按电源键确定  
选中yes，单按电源键确定即可恢复出厂设置
VR重启后戴着头显根据vr界面提示继续操作，直到连wifi的那一步，
```

```
vr连接一个墙外的wifi access point：戴着头显等，直到只显示已连接 3个字，才可以点底部完成按钮

手机打开加速器

热点不要断

使用事先注册好的meta账户登录
https://mvxhi34zzhs.feishu.cn/docx/OzuZd88uzo2D5xxjQTNcsGEwnxh

XR更新2个GB左右后，出现5位验证码，使用meta horizon配对

手机app点击设备-头戴设备设置-开发者模式-打开
```