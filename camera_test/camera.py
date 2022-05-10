#!/usr/bin/env python3
# encoding: utf-8
import cv2
import numpy as np
import LineDetectionPlugin as LD


import sys
import rospy
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float64
from std_srvs.srv import SetBool, SetBoolResponse


fourcc = cv2.VideoWriter_fourcc(*'XVID')
outvideo = cv2.VideoWriter("out.avi",fourcc, 30.0, (640,480),True)

cap = cv2.VideoCapture()

def camera_switch_turn(req):
    print("camera operate req received: %s" %(req.data))
    if req.data:
        global cap
        cap = cv2.VideoCapture("./zoulang.mp4")
    else:
        cap.release()

    if cap.isOpened():
        print("camera now opened")
    else:
        print("camera now closed")
    return SetBoolResponse(True, "ok")

if __name__ == '__main__':

    rospy.init_node('camera', anonymous=True)
    theta_pub = rospy.Publisher('visual_track/bottomRight_X', Float64, queue_size=1)
    cam_switch_srv = rospy.Service('visual_track/enable', SetBool, camera_switch_turn)
    
    n = 1
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        while cap.isOpened(): 
            print(n)
            n = n+ 1
            ret,frame = cap.read()
            '''
            print("main frame")
            print(frame)
            print("end read frame")
            '''
            if frame is None:
                break

            bottompointleft, slopleft, bottompointright, slopright, frame = LD.refineprocess(frame)
            msg = Float64()
            if len(bottompointright) == 0:
                msg.data = -1.0
                theta_pub.publish(msg)
            else:
                msg.data = bottompointright[0]
                theta_pub.publish(msg)
            

            # 显示并保存结果
            outvideo.write(frame)
            cv2.imshow("result", frame)
            cv2.waitKey(1)
            rate.sleep()
        
        # 相机未开，等待服务
        rate.sleep()

    rospy.spin()
    cap.release()
    outvideo.release()
