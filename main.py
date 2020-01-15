import numpy as np
import argparse
import cv2
import time
from util import *
from synthetic import computeDisp_syn
from real import computeDisp
from refinement import *

parser = argparse.ArgumentParser(description='Disparity Estimation')
parser.add_argument('--input-left', default='./data/Synthetic/TL0.png', type=str, help='input left image')
parser.add_argument('--input-right', default='./data/Synthetic/TR0.png', type=str, help='input right image')
parser.add_argument('--output', default='./TL0.pfm', type=str, help='left disparity map')


# You can modify the function interface as you like
def computeDisp_comb(Il, Ir, mode):
    h, w, ch = Il.shape
    if mode:
        disp = computeDisp(Il, Ir, search_depth=15, occlusion_cost=-1, max_iterations=4,census_kernel_size=7, dissim_method='census')
        return disp
    else:
        disp = computeDisp_syn(Il, Ir)
        return disp


def main():
    args = parser.parse_args()

    print(args.output)
    print('Compute disparity for %s' % args.input_left)
    img_left = cv2.imread(args.input_left)
    img_right = cv2.imread(args.input_right)
    tic = time.time()
    mode = judgement(img_left,img_right)
    disp = computeDisp_comb(img_left, img_right, mode)
    toc = time.time()
    #cv2.imwrite('output.jpg', disp)
    writePFM(args.output, disp)
    print('Elapsed time: %f sec.' % (toc - tic))
    #GT_path = args.input_left[0:19]+'D'+args.input_left[19]+'.pfm'
    #GT = readPFM(GT_path)
    #print('score:',cal_avgerr(GT, disp))


if __name__ == '__main__':
    main()
