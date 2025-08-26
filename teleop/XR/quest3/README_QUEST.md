# Meta quest3 usage
## setup
1. local tunning
   - open up the terminal and run `npm install -g localtunnel`
   - run `lt --port 8000`, and you will get a website ip, access it and set the password, mine is https://tough-symbols-vanish.loca.lt; passwd: zyx_hirol123 


2. PC setup
   - install adb by 
       ```
           sudo apt-get update
           sudo apt-get install android-tools-adb
       ```
   - check whether installed successfully by `adb version`
   - connect your xr to the pc by usb and check your vr device by `sudo adb devices` and you will see 
   ```
       List of devices attached
       2G0YC5ZGC900XR	unauthorized
   ```
   - If unauthorized appears, it means the XR device has not been authorized. Resatrt the xr device and put on the XR device and click "Allow" in the pop-up window asking for USB debugging permission. Then execute the command again
   ```
       List of devices attached
       2G0YC5ZGC900XR	device
   ```
   - run `sudo adb -s 2G0YC5ZGC900XR reverse tcp:8012 tcp:8012`
   - verify the result by `sudo adb -s 2G0YC5ZGC900XR reverse --list`, output is" `UsbFfs tcp:8012 tcp:8012
   `
   - generate cert and key by `openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout key.pem -out cert.pem`
   - wireless setting: 
    ```
        adb tcpip 5555
        sudo adb shell ifconfig wlan0
        adb connect <device ip>:5555
    ```
   - check whether connected successfuly by `adb devices -l`
   - reverse port for ip: `adb -s <device_ip>:5555 reverse tcp:8012 tcp:8012`, check by `adb -s <device_ip>:5555 reverse --list`

3. running of the testing scipt
 - `python test/test_xr_duo_fr3.py`
 - wear the xr glass and connect the wifi to the hirol
 - open the webpage of `https://<your pc ip>:8012?ws=wss://<your pc ip>:8012`, the pc ip should be same with your pc running the script
  

4. Notice for the device in hirol lab
- device ip: `192.168.171.56`
- device number: `2G0YC5ZGC900XR`
  