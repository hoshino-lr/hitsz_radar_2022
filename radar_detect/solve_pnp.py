"""
预警类
created by 黄继凡 2021/1
最新修改 by 黄继凡 2022/1/14
"""
import cv2
import numpy as np

objPoints = np.array([[4021, 3994, 0],# R4下来的角
                      [7820, 4550, 0], #坡除角
                      [6756, 6300, 600], #坡上角
                      [2250, 8311, 1376], #烧饼轨道 u远
                      [2250,4825,1376], #烧饼轨道 近
                      [6970, 8311,600]], dtype=np.float32) #坡，另一边角
# 对象坐标点，此处依次使用定位点坐标1:(7740,4515,0)、
# 2:(8880,5260,0)、6:(7710,9120,600)、7:(2720,9450,1376)
objPoints = np.array([[4021, 3994, 0],# R4下来的角
                      [7820, 4550, 0], #坡除角
                      [6756, 6300, 600], #坡上角
                      [2250, 8311, 1376], #烧饼轨道 u远
                      [2250,4825,1376], #烧饼轨道 近
                      [6970, 8311,600]], dtype=np.float32) #坡，另一边角

imgPoints = np.zeros((6, 2), dtype=np.float32)
rvec = np.zeros((3, 1), dtype=np.float64)
tvec = np.zeros((3, 1), dtype=np.float64)
# 鼠标回调事件
count = 0  # 计数，依次确定个点图像坐标


def select_callback(event, x, y, flags, param):
    global count
    global imgPoints
    if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键开始输入imgPoints
        imgPoints[count, :] = [float(x), float(y)]
        if count != 5:
            count += 1
        print("the coordinate (x,y) is", x, y)
    if event == cv2.EVENT_MBUTTONDOWN:  # 鼠标中键按下重置imgPoints
        imgPoints = np.zeros((6, 2))
        count = 0


# 四点标定函数
def locate_pick():
    cv2.setMouseCallback("PNP", select_callback)
    while (True):
        key = cv2.waitKey(0)
        if key == 13:  # 按下回车键输出rvec、tvec
            u,rvec,tvec,inliers =cv2.solvePnPRansac(objectPoints=objPoints,
                                                    distCoeffs=distCoeffs,
                                                    cameraMatrix=cameraMatrix,
                                                    imagePoints=imgPoints)
            break
    print(rvec,tvec)
    return rvec, tvec


if __name__ == "__main__":
    PIC = cv2.imread("/home/hoshino/CLionProjects/hitsz_radar/resources/beijing.png")
    cv2.namedWindow("PNP", cv2.WINDOW_NORMAL)
    cv2.imshow("PNP", PIC)
    rvec, tvec = locate_pick()
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]  # 旋转向量转化为旋转矩阵
    T[:3, 3] = tvec.reshape(-1)  # 加上平移向量
    T = np.linalg.inv(T)  # 矩阵求逆
    print(T, (T @ (np.array([0, 0, 0, 1])))[:3])
    cv2.destroyAllWindows()
