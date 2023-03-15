# coding=utf-8
# 人脸图片处理流程:
# 灰度化
# 图片resize->(160, 160)
# 不同程度、类型的模糊化操作
# 不同类型的模糊检测算法
# 统计结果

# 全身人图片处理流程:
# 灰度化
# 图片resize->(128, 256)
# 不同程度、类型的模糊化操作
# 不同类型的模糊检测算法
# 统计结果

import cv2


def imgPreHandle(img, resize_size=(160, 160)):
    img_cp = img.copy()
    # resize img
    out = cv2.resize(img_cp, resize_size, interpolation=cv2.INTER_CUBIC)
    return out


def blur(img, blur_type="resize", blur_degree=1):
    # different type blur method
    img_cp = img.copy()
    if isinstance(blur_degree, int) and blur_degree > 0:
        kernal_size = (3 * blur_degree, 3 * blur_degree)
        if blur_type == "mean":
            blured_img = cv2.boxFilter(img_cp, -1, (3, 3), normalize=True)
        elif blur_type == "gaussian":
            blured_img = cv2.GaussianBlur(img_cp, kernal_size, sigmaX=0)
        elif blur_type == "median":
            blured_img = cv2.medianBlur(img_cp, kernal_size)
        elif blur_type == "resize":
            h, w = img_cp.shape[:2]
            reduced_w, reduced_h = int(w / blur_degree), int(h / blur_degree)
            blured_img = cv2.resize(img_cp, (reduced_w, reduced_h), cv2.INTER_CUBIC)
        else:
            raise NameError("No such blur_type")
        return blured_img
    else:
        return img_cp

