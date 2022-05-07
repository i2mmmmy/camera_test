#!/usr/bin/env python3.8
# encoding: utf-8
import cv2
import numpy as np
import LineDetectionPlugin as LD

import rospy
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float64


fourcc = cv2.VideoWriter_fourcc(*'XVID')
outvideo = cv2.VideoWriter("out.avi",fourcc, 30.0, (640,480),True)




if __name__ == '__main__':
    #cap = cv2.VideoCapture("./Data/WeChat_20220402195554.mp4")
    #cap = cv2.VideoCapture("./Data/WeChat_20220402195624.mp4")
    #cap = cv2.VideoCapture("./Data/WeChat_20220402195648.mp4")
    rospy.init_node('talker', anonymous=True)
    theta_pub = rospy.Publisher('bottomRight_X', Float64, queue_size=1)
    #cap = cv2.VideoCapture(1)
    # cap = cv2.VideoCapture("./zoulang.mp4")
    cap = cv2.VideoCapture(4)

    n = 1
    rate = rospy.Rate(10) # 10hz

    while cap.isOpened() and not rospy.is_shutdown():
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
        # print("bottommmmmmmmmmmmmmmmmmm:", type(bottompointleft))
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

    cap.release()
    outvideo.release()
