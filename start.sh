#!/bin/bash
cd /home/mark/hitsz_radar
echo " "|sudo -S chmod 777 /dev/ttyACM0
gnome-terminal -t "roscore" -x bash -c "roscore;exec bash;"
sleep 1s
gnome-terminal -t "ros_server" -x bash -c "roslaunch livox_ros_driver livox_lidar.launch;exec bash;"
sleep 1s
gnome-terminal -t "main.py" -x bash -c "python3 main_v2.py;exec bash;"
