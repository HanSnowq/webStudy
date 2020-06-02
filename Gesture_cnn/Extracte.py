# encoding:utf-8
import cv2
import numpy as np


def get_image(path):  # 获取图片,灰度
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def viewImage(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 480, 600);
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Gaussian_Blur(gray):  # 高斯去噪(去除图像中的噪点)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    return blurred


def erode_dilate(img):  # 腐蚀膨胀获取边界
    kernel = np.ones((5, 5), np.uint8)
    grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    dilation = cv2.dilate(grad, kernel, iterations=1)
    return grad, dilation


def Thresh_and_blur(gradient):  # 设定阈值
    blurred = Gaussian_Blur(gradient)
    (_, thresh) = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
    return thresh


def image_morphology(thresh):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))  # 腐蚀膨胀去噪
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)  # 突出边界
    closed = cv2.dilate(closed, None, iterations=4)
    return closed


def findcnts_and_box_point(closed):  # 计算最大轮廓的旋转包围盒
    contours, hierarchy = cv2.findContours(closed.copy(),
                                           cv2.RETR_LIST,  # 外侧信息
                                           cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    return box


def drawcnts_and_cut(original_img, box):
    draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    if (x1 < 0):
        x1 = 0
    x2 = max(Xs)
    y1 = min(Ys)
    if (y1 < 0):
        y1 = 0
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    crop_img = original_img[y1:y1 + hight, x1:x1 + width]
    return draw_img, crop_img


def change_image(img_path):
    img, gray = get_image(img_path)
    blurred = Gaussian_Blur(gray)
    gradient, dil = erode_dilate(blurred)
    thresh = Thresh_and_blur(gradient)
    closed = image_morphology(thresh)
    box = findcnts_and_box_point(closed)
    draw_img, crop_img = drawcnts_and_cut(img, box)
    viewImage('original_img', img)
    viewImage('GaussianBlur', blurred)
    viewImage('gradient', gradient)
    viewImage('dil', dil)
    viewImage('thresh', thresh)
    viewImage('closed', closed)
    viewImage('draw_img', draw_img)
    viewImage('crop_img', crop_img)
    cv2.imwrite(img_path, crop_img)

