# encoding: utf-8
import cv2
import numpy as np


w = 640
h = 480
startp = (-1,-1)

cntthresh = 100

init = 1
mc_history = startp
hisnum = 30 #走的慢可以做设置一些帧，或者帧率比较快的情况下
stahisnum = 15
history_result = []
dist_his = []
distthresh = 50
clearthresh = 100



def result_filter(mc):
    global stahisnum
    isre = False
    history_result.append(mc)
    num = len(history_result)
    if num > 1:
        tempdist = np.sqrt((history_result[num-1][0]-history_result[num-2][0])**2 \
                           + (history_result[num-1][1]-history_result[num-2][1])**2)
        dist_his.append(tempdist)

    if num<hisnum:
        return isre,mc
    numdis = num -2

    if dist_his[numdis] > clearthresh:
        history_result.clear()
        dist_his.clear()
        return isre,mc

    for s in range(stahisnum):
        if dist_his[numdis-s] > distthresh:
            return isre,mc

    temppoint = [0,0]
    for s in range(stahisnum):
        temppoint[0] = temppoint[0] + history_result[num-1-s][0]
        temppoint[1] = temppoint[1] + history_result[num - 1 - s][1]

    mc[0] = temppoint[0] / float(stahisnum)
    mc[1] = temppoint[1] / float(stahisnum)
    print(mc)
    isre = True

    return isre,mc


def processaside(img,isright):
    # bottomspace = 30

    bottompoint  = []
    slopre = []

    points = np.where(img > 0)
    if len(points[0]) == 0: # 可能没有这条线
        return bottompoint, slopre


    coord = list(zip(points[1], points[0]))
    num,pp = np.array(coord).shape
    cont = np.resize(coord,(num,1,pp))

    #TODO:检测直线斜率的一致性，检测直线斜率位置的一致性，加入约束

    hull = cv2.convexHull(cont)
    length = len(hull)
    slopall = []
    ball = []
    templine = []
    for i in range(len(hull)):
        x1,y1 = hull[i][0]
        x2,y2 = hull[(i + 1) % length][0]
        '''
        test = ((x1-x2)^2 + (y1-y2)^2)
        s1 = (x1-x2)^2
        s2 = (y1-y2)^2
        print("x1:", x1, " x2:", x2, " y1:", y1, " y2:", y2)
        print("s1:", s1, " s2:", s2)
        print("test:", test)
        '''
        if np.sqrt((x1-x2)^2 + (y1-y2)^2) < 20:
            continue

        if y1 < h/5*4 and y2 < h/5*4:
            continue

        if abs(x1 - x2) < 5 or abs(y1 - y2) < 5:  # 对于斜率为0和无穷大的情况进行剔除
            continue

        fit = np.polyfit((x1, x2), (y1, y2), 1)  # 拟合成直线
        slope = fit[0]  # 斜率
        #print(slope)

        if np.abs(slope) < 5 and np.abs(slope) > 1.0:
            #cv2.line(img, tuple(hull[i][0]), tuple(hull[(i + 1) % length][0]), (0, 255, 0), 2)

            if y1 == h-1 or y2 == h-1:
                continue

            if isright and slope < 0: # 大于零的在右边
                continue

            if (not isright) and slope > 0: # 小于零的在左边
                continue

            print("result: ", ((x1 - x2) ^ 2))
            if np.sqrt((x1 - x2) ^ 2 + (y1 - y2) ^ 2) < 100: #对边长进行筛选
                continue

            templine.append([x1,y1,x2,y2])
            slopall.append(fit[0])
            ball.append(fit[1])

            tempy = int(h-1)
            tempx = int((tempy - fit[1])/fit[0])

            cv2.line(img, (tempx,tempy), tuple(hull[(i + 1) % length][0]), (255, 255, 255), 3)
            cv2.line(img, (tempx,tempy),tuple(hull[i][0]),(255, 255, 255), 3)

    dist = []
    #print(slopall)
    num = len(slopall)

    if num > 2: #至少是3，才能完成投票
        for tempslop in slopall:
            tempdist =np.sum(abs(np.array(slopall)-tempslop))/(num-1)
            dist.append(tempdist)
            #print(tempslop,tempdist)

    for i in range(0,len(slopall)):
        if len(dist) == len(slopall):
            if dist[i] < 0.10:
                tempy = int(h - 1)
                tempx = int((tempy - ball[i]) / slopall[i])

                cv2.line(img, (tempx, tempy), (templine[i][0], templine[i][1]), (0, 255, 255), 1)
                cv2.line(img, (tempx, tempy), (templine[i][2], templine[i][3]), (0, 255, 255), 1)
        else:
            tempy = int(h-1)
            tempx = int((tempy - ball[i])/slopall[i])

            cv2.line(img, (tempx, tempy), (templine[i][0], templine[i][1]), (0, 255, 255), 1)
            cv2.line(img, (tempx, tempy), (templine[i][2], templine[i][3]), (0, 255, 255), 1)


    # last row process
    temprowmat = img[h-1,:,1]
    temppoints = cv2.findNonZero(temprowmat)
    if temppoints is None:
        return bottompoint, slopre

    temppoints = temppoints.reshape(temppoints.shape[0],temppoints.shape[2])

    #去掉孤立点
    disttmp = []
    num = len(temppoints)
    if num > 2:
        for point in temppoints:
           disttmp.append(np.sum(abs(np.array(temppoints[:,1]) - point[1])) / num)
           #print(disttmp)
        yespoint = []
        for i in range(0,num):
            #print(num,len(disttmp))
            if disttmp[i] < 100:
                yespoint.append(temppoints[i])

        temppoints = yespoint

    if len(temppoints) == 0:
        return bottompoint, slopre

    if isright:
        # 找到最小的
        temppos = np.min(temppoints,axis=0)
        bottompoint = [temppos[1],h-1]
        #print("right : ",temppos)
    else:
        temppos = np.max(temppoints,axis=0)
        bottompoint = [temppos[1],h-1]
        #print("left : ",temppoints)

    # search another point for slop
    temprowmat = img[h-30,:,1]
    temppoints = cv2.findNonZero(temprowmat)
    if temppoints is None:
        return bottompoint, slopre
    temppoints = temppoints.reshape(temppoints.shape[0],temppoints.shape[2])

    #去掉孤立点
    disttmp = []
    num = len(temppoints)
    if num > 2:
        for point in temppoints:
           disttmp.append(np.sum(abs(np.array(temppoints[:,1]) - point[1])) / num)
           #print(disttmp)
        yespoint = []
        for i in range(0,num):
            if disttmp[i] < 100:
                yespoint.append(temppoints[i])

        temppoints = yespoint

    if len(temppoints) == 0:
        return bottompoint, slopre

    if isright:
        # 找到最小的
        temppos = np.min(temppoints,axis=0)
        tempbottompoint = [temppos[1],h-30]
        #print("right : ",temppos)
    else:
        temppos = np.max(temppoints,axis=0)
        tempbottompoint = [temppos[1],h-30]
        #print("left : ",temppoints)

    cv2.circle(img, (bottompoint[0],bottompoint[1]), 3, (0, 0, 255), 0)
    cv2.circle(img, (tempbottompoint[0],tempbottompoint[1]), 3, (0, 0, 255), 0)

    fitend = np.polyfit((bottompoint[0],tempbottompoint[0]),(bottompoint[1],tempbottompoint[1]),
                        1)
    #slop = np.arctan(fitend[0])
    slopre.append(fitend[0])

    cv2.imshow("side image",img)
    cv2.waitKey(1)

    return bottompoint,slopre


def prolines(dlines,img0):
    global cntthresh
    global mc_history
    global init
    global road
    redlines = []
    mc = []

    h,w,c = img0.shape
    roih = h/2 + 1.2
    edgeimg = np.zeros_like(img0)
    halfline = w/2
    for line in dlines:

        for x1, y1, x2, y2 in line: #只有线头和线尾

            if abs(x1-x2)<5 or abs(y1-y2)<5: #对于斜率为0和无穷大的情况进行剔除
                continue

            fit = np.polyfit((x1, x2), (y1, y2), 1)  # 拟合成直线
            slope = fit[0]  # 斜率

            if np.abs(slope) < 5 and np.abs(slope) > 1.0: # 通过slope>0 和 slope<0来区分左右线
                x0 = int(round(line[0][0]))
                y0 = int(round(line[0][1]))
                x1 = int(round(line[0][2]))
                y1 = int(round(line[0][3]))

                #这里加上如果是左边的边，则不能是右边边的斜率；
                #如果是右边的边，则不能是左边的边的斜率
                #基于假设，线不过中线；
                if slope < 0 : #应该在左边
                    if x0 > halfline or x1 > halfline:
                        continue
                else: # 应该在右边
                    if x0 < halfline or x1 < halfline:
                        continue

                cv2.line(edgeimg, (x0, y0), (x1, y1), (255,255,255), 1, cv2.LINE_AA)
                redlines.append(line)

    edgeimg[0:int(roih),:,:] = 0

    contourss, _ = cv2.findContours(edgeimg[:,:,0], cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)  # cv2.RETR_EXTERNAL 定义只检测外围轮廓
    edgeimg = np.zeros_like(edgeimg)
    for cnt in contourss:
        #print(cnt.shape)
        if cv2.contourArea(cnt) < cntthresh:
            continue
        cv2.drawContours(edgeimg, cnt, -1, (255, 255, 255), 2)

    midium = 30
    maskleft = np.zeros_like(edgeimg)
    maskleft[:,0:int(halfline-midium),:] = 1
    maskleft = maskleft * edgeimg

    maskright = np.zeros_like(edgeimg)
    maskright[:,int(halfline+midium):-1,:] = 1
    maskright = maskright * edgeimg

    # cv2.imshow("maskleft", maskleft)
    # cv2.imshow("maskrightt",maskright)
    # cv2.waitKey(1)

    bottompointleft,slopleft = processaside(maskleft,0)
    bottompointright, slopright = processaside(maskright,1)

    print(bottompointleft,slopleft,bottompointright,slopright)

    #从此得到左右两边的两条线，或者是一条线或者是一边的线也要考虑 : 左右两边的分别求，基于假设，线不过中线
    # 线连续的要考虑，线不连续的也要考虑：发现最下面一行没有符合要求点的时候，要延长符合条件的边。

    # cv2.imshow("edgeimg",edgeimg)
    # cv2.waitKey(1)
    return bottompointleft,slopleft,bottompointright,slopright

def FLDtest(img0):
    import cv2
    import numpy as np

    img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

    # 创建一个LSD对象
    fld = cv2.ximgproc.createFastLineDetector()
    # 执行检测结果
    dlines = fld.detect(img)
    # 绘制检测结果
    # rawn_img = fld.drawSegments(img0,dlines, )
    # cv2.imshow("raw result",rawn_img)
    # cv2.waitKey(1)
    # 对检测出的边缘线进行筛选
    bottompointleft,slopleft,bottompointright,slopright = prolines(dlines,img0)
    # 对检测出的点，多帧之间进行处理，保持稳定性；
    #ret,mc = result_filter(mc)
    # 画出检测到的点
    # if ret:
    #     cv2.circle(img0,(int(mc[0]),int(mc[1])), 10,(255,0,0),3,1)
    # 对筛选之后的线画出来
    # re1 = np.zeros_like(img0)
    # for dline in relines:
    #     x0 = int(round(dline[0][0]))
    #     y0 = int(round(dline[0][1]))
    #     x1 = int(round(dline[0][2]))
    #     y1 = int(round(dline[0][3]))
    #     cv2.line(img0, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)
    #     cv2.line(re1,(x0, y0), (x1, y1),(255),1,cv2.LINE_AA)
    #
    # cv2.imshow("re1",re1)
    # cv2.waitKey(1)

    return bottompointleft,slopleft,bottompointright,slopright


history_r = []
history_r_num = 150

def refineprocess(frame):

    bottompointleft, slopleft, bottompointright, slopright = FLDtest(frame)

    if len(history_r) < history_r_num-1:
        history_r.append([bottompointleft, slopleft, bottompointright, slopright])


    if len(history_r) == history_r_num - 1:

        displ = 0
        cpl = 0
        dissl = 0
        csl = 0
        dispr = 0
        cpr = 0
        dissr = 0
        csr = 0
        for pl,sl,pr,sr in history_r:
            if len(bottompointleft) > 0 and len(pl)>0:
                displ = displ + abs(pl[0]-bottompointleft[0])
                cpl = cpl + 1
            if len(slopleft) > 0 and len(sl) > 0:
                dissl = dissl + abs(slopleft[0]-sl[0])
                csl = csl + 1
            if len(bottompointright) > 0 and len(pr) > 0:
                dispr = dispr + abs(bottompointright[1]-pr[1])
                cpr = cpr + 1
            if len(slopright) >0 and len(sr) >0:
                dissr = dissr + abs(slopright[0]-sr[0])
                csr = csr + 1


        if csl > 3 and cpl > 3:
            if displ/cpl >30 or dissl/csl > 0.5:
                print("_____________________________________----------------------------")
                bottompointleft = []
                slopleft = []

        if cpr > 3 and csr > 3:
            if dispr/cpr > 30 or dissr/csr > 0.5:
                print("_____________________________________----------------------------")
                bottompointright = []
                slopleft = []

    history_r.append([bottompointleft, slopleft, bottompointright, slopright])
    print("bottompointright")
    print(bottompointright)
    print("print bottompointright")
    del history_r[0]

    if len(bottompointleft) > 1 and len(slopleft) > 0:
        cv2.circle(frame, bottompointleft, 3, (0, 0, 255), 2)
        b = bottompointleft[1] - slopleft[0] * bottompointleft[0]
        ytemp = frame.shape[0] - 100
        xtemp = (ytemp - b) / slopleft[0]
        cv2.line(frame, bottompointleft, (int(xtemp), int(ytemp)), (0, 255, 0), 2)
    if len(bottompointright) > 1 and len(slopright) > 0:
        cv2.circle(frame, bottompointright, 3, (0, 0, 255), 2)
        b = bottompointright[1] - slopright[0] * bottompointright[0]
        ytemp = frame.shape[0] - 100
        xtemp = (ytemp - b) / slopright[0]
        cv2.line(frame, bottompointright, (int(xtemp), int(ytemp)), (0, 255, 0), 2)

    return bottompointleft, slopleft, bottompointright, slopright,frame


