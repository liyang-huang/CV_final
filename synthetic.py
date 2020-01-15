import numpy as np
import scipy.ndimage
import cv2
import matplotlib.pyplot as plt
from numpy.linalg import norm
import sys


def compute_data_cost(I1, I2, num_disp_values, Tau):
    """data_cost: a 3D array of sixe height x width x num_disp_value;
    data_cost(y,x,l) is the cost of assigning the label l to pixel (y,x).
    The cost is min(1/3 ||I1[y,x]-I2[y,x-l]||_1, Tau)."""
    h,w,_ = I1.shape
    dataCost=np.zeros((h,w,num_disp_values))

    for lp in range(num_disp_values):
        tmp = np.roll(I2, lp, axis=1)
        tmp[:, :lp] = I2[:, :lp]
        dataCost[:, :, lp] = np.minimum(1./3*norm(I1 - tmp, axis=2, ord=1), Tau*np.ones((h, w)))

    return dataCost

def compute_energy(dataCost,disparity,Lambda):
    """dataCost: a 3D array of sixe height x width x num_disp_values;
    dataCost(y,x,l) is the cost of assigning the label l to pixel (y,x).
    disparity: array of size height x width containing disparity of each pixel.
    (an integer between 0 and num_disp_values-1)
    Lambda: a scalar value.
    Return total energy, a scalar value"""
    h,w,num_disp_values = dataCost.shape

    hh, ww = np.meshgrid(range(h), range(w), indexing='ij')
    dplp = dataCost[hh, ww, disparity]

    # Unitary cost of assigning this disparity to each pixel
    energy = np.sum(dplp)

    # Compute interaction cost of each neighbors
    interactionCostU = Lambda*(disparity - np.roll(disparity, 1, axis=0) != 0)
    interactionCostL = Lambda*(disparity - np.roll(disparity, 1, axis=1) != 0)
    interactionCostD = Lambda*(disparity - np.roll(disparity, -1, axis=0) != 0)
    interactionCostR = Lambda*(disparity - np.roll(disparity, -1, axis=1) != 0)

    # Ignoring edge costs
    interactionCostU[0, :] = 0
    interactionCostL[:, 0] = 0
    interactionCostD[-1, :] = 0
    interactionCostR[:, -1] = 0

    # Adding interaction cost of each neighbors
    energy += np.sum(interactionCostU)
    energy += np.sum(interactionCostL)
    energy += np.sum(interactionCostD)
    energy += np.sum(interactionCostR)

    return energy

def update_msg(msgUPrev,msgDPrev,msgLPrev,msgRPrev,dataCost,Lambda):
    """Update message maps.
    dataCost: 3D array, depth=label number.
    msgUPrev,msgDPrev,msgLPrev,msgRPrev: 3D arrays (same dims) of old messages.
    Lambda: scalar value
    Return msgU,msgD,msgL,msgR: updated messages"""
    msgU=np.zeros(dataCost.shape)
    msgD=np.zeros(dataCost.shape)
    msgL=np.zeros(dataCost.shape)
    msgR=np.zeros(dataCost.shape)

    h,w,num_disp_values = dataCost.shape

    msg_incoming_from_U = np.roll(msgDPrev, 1, axis=0)
    msg_incoming_from_L = np.roll(msgRPrev, 1, axis=1)
    msg_incoming_from_D = np.roll(msgUPrev, -1, axis=0)
    msg_incoming_from_R = np.roll(msgLPrev, -1, axis=1)
    
    # msg_incoming_from_U[0, :] = 0
    # msg_incoming_from_L[:, 0] = 0
    # msg_incoming_from_D[-1, :] = 0
    # msg_incoming_from_R[:, -1] = 0

    npqU = dataCost + 0.5*(msg_incoming_from_L + msg_incoming_from_D + msg_incoming_from_R)
    npqL = dataCost + 0.5*(msg_incoming_from_U + msg_incoming_from_D + msg_incoming_from_R)
    npqD = dataCost + 0.5*(msg_incoming_from_L + msg_incoming_from_U + msg_incoming_from_R)
    npqR = dataCost + 0.5*(msg_incoming_from_L + msg_incoming_from_D + msg_incoming_from_U)

    spqU = np.amin(npqU, axis=2)
    spqL = np.amin(npqL, axis=2)
    spqD = np.amin(npqD, axis=2)
    spqR = np.amin(npqR, axis=2)

    for lp in range(num_disp_values):
        msgU[:, :, lp] = np.minimum(npqU[:, :, lp], Lambda + spqU)
        msgL[:, :, lp] = np.minimum(npqL[:, :, lp], Lambda + spqL)
        msgD[:, :, lp] = np.minimum(npqD[:, :, lp], Lambda + spqD)
        msgR[:, :, lp] = np.minimum(npqR[:, :, lp], Lambda + spqR)

    return msgU,msgD,msgL,msgR

def normalize_msg(msgU,msgD,msgL,msgR):
    """Subtract mean along depth dimension from each message"""

    avg=np.mean(msgU,axis=2)
    msgU -= avg[:,:,np.newaxis]
    avg=np.mean(msgD,axis=2)
    msgD -= avg[:,:,np.newaxis]
    avg=np.mean(msgL,axis=2)
    msgL -= avg[:,:,np.newaxis]
    avg=np.mean(msgR,axis=2)
    msgR -= avg[:,:,np.newaxis]

    return msgU,msgD,msgL,msgR

def compute_belief(dataCost,msgU,msgD,msgL,msgR):
    """Compute beliefs, sum of data cost and messages from all neighbors"""
    beliefs=dataCost.copy()

    msg_incoming_from_U = np.roll(msgD, 1, axis=0)
    msg_incoming_from_L = np.roll(msgR, 1, axis=1)
    msg_incoming_from_D = np.roll(msgU, -1, axis=0)
    msg_incoming_from_R = np.roll(msgL, -1, axis=1)

    beliefs += 0.5*(msg_incoming_from_D + msg_incoming_from_L + msg_incoming_from_R + msg_incoming_from_U)

    return beliefs

def MAP_labeling(beliefs):
    """Return a 2D array assigning to each pixel its best label from beliefs
    computed so far"""
    return np.argmin(beliefs, axis=2)

def stereo_bp(I1,I2,num_disp_values,Lambda,Tau=15,num_iterations=60):
    """The main function"""
    dataCost = compute_data_cost(I1, I2, num_disp_values, Tau)
    energy = np.zeros((num_iterations)) # storing energy at each iteration
    # The messages sent to neighbors in each direction (up,down,left,right)
    h,w,_ = I1.shape
    msgU=np.zeros((h, w, num_disp_values))
    msgD=np.zeros((h, w, num_disp_values))
    msgL=np.zeros((h, w, num_disp_values))
    msgR=np.zeros((h, w, num_disp_values))

    # print('Iteration (out of {}) :'.format(num_iterations))
    for iter in range(num_iterations):
        # print('\t'+str(iter), end='')
        msgU,msgD,msgL,msgR = update_msg(msgU,msgD,msgL,msgR,dataCost,Lambda)
        msgU,msgD,msgL,msgR = normalize_msg(msgU,msgD,msgL,msgR)
        # Next lines unused for next iteration, could be done only at the end
        beliefs = compute_belief(dataCost,msgU,msgD,msgL,msgR)
        disparity = MAP_labeling(beliefs)
        energy[iter] = compute_energy(dataCost,disparity,Lambda)

    return disparity,energy

def refine(disparityL):
    h, w = disparityL.shape
    
    hole = 12
    maxv = 45
    for i in range(h):
        for j in range(w):
            if disparityL[i][j] < hole or disparityL[i][j] > maxv:
                left = w
                right = w
                k = j+1
                while k < w and disparityL[i][k] < hole:
                    k += 1
                if k < w:
                    right = disparityL[i][k]
                k = j-1
                while k >= 0 and disparityL[i][k] < hole:
                    k -= 1
                if k >= 0:
                    right = disparityL[i][k]
                disparityL[i][j] = min(left, right)
    return disparityL

def hisEqulColor(img):
    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    channels=cv2.split(ycrcb)
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
    return img

def computeDisp_syn(img_left, img_right):
    # Parameters
    num_disp_values=50 
    Lambda=1
    nb_iterations=10

    # Input
    img_left = hisEqulColor(img_left)
    img_right = hisEqulColor(img_right)

    # Convert as float gray images
    img_left = img_left.astype(float)
    img_right = img_right.astype(float)

    # Gaussian filtering
    I1=scipy.ndimage.filters.gaussian_filter(img_left, 0.3)
    I2=scipy.ndimage.filters.gaussian_filter(img_right, 0.3)

    disparity,energy = stereo_bp(I1,I2,num_disp_values,Lambda, num_iterations=nb_iterations,Tau=15)
    
    disL = disparity.copy()
    disp = refine(disL)    
    disp = cv2.ximgproc.weightedMedianFilter(joint=img_left.astype(np.uint8), 
            src=disp.astype(np.uint8), r=30, sigma=25)
    
    return disp.astype(np.float32)
