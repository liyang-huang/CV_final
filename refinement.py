import sys
import numpy as np
import cv2
#import cv2.ximgproc as xip


def search_r(src, index):
    h, w = src.shape
    id_x, id_y = index
    lb = id_x+1
    if lb >= w:
        lb = w
    for x in range(lb,w):
        if src[id_y,x] >= 0:
            return src[id_y,x]
    return sys.maxsize

def search_l(src, index):
    h, w = src.shape
    id_x, id_y = index
    lb = id_x - 1
    if lb <= 0:
        lb = 0
    for x in range(lb,0,-1):
        if src[id_y,x] >= 0:
            return src[id_y, x]
    return sys.maxsize


def hole_filling(src):
    h, w = src.shape
    for y in range(h):
        for x in range(w):
            if src[y,x] < 0 :
                src[y, x] = min(search_l(src,(x,y)),search_r(src,(x,y)))

def judgement(Il,Ir):

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(Il, None)
    kp2, des2 = sift.detectAndCompute(Ir, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    # ratio test as per Lowe's paper
    for m,n in matches:
        if m.distance < 0.3*n.distance:
            good.append(m)

    pts1 = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    pts2 = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    index = int(0)
    match_count = np.count_nonzero( pts1[:,0,0] > pts2[:,0,0] )
    #print(match_count,pts1.shape[0])
    return (pts1.shape[0] - match_count) > 7
