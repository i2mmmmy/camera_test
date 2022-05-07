# encoding: utf-8
import cv2
import numpy as np
import LineDetection as LD

fourcc = cv2.VideoWriter_fourcc(*'XVID')
outvideo = cv2.VideoWriter("out.avi",fourcc, 30.0, (640,480),True)

if __name__ == '__main__':
    #cap = cv2.VideoCapture("./Data/WeChat_20220402195554.mp4")
    #cap = cv2.VideoCapture("./Data/WeChat_20220402195624.mp4")
    #cap = cv2.VideoCapture("./Data/WeChat_20220402195648.mp4")

    #cap = cv2.VideoCapture(1)
    # cap = cv2.VideoCapture("./zoulang.mp4")
    cap = cv2.VideoCapture(4)

    n = 1
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


        # 显示并保存结果
        outvideo.write(frame)
        cv2.imshow("result", frame)
        cv2.waitKey(1)



    cap.release()
    outvideo.release()
