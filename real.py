import numpy as np
import argparse
import cv2
import time
from util import writePFM
import graph 
import refinement

#TYPE = 'Synthetic'
#TYPE = 'Real'
types = ['Real']

parser = argparse.ArgumentParser(description='Disparity Estimation')
parser.add_argument('--input-left', default='./data/Real/TL0.bmp', type=str, help='input left image')
parser.add_argument('--input-right', default='./data/Real/TR0.bmp', type=str, help='input right image')
parser.add_argument('--output', default='./TL0.pfm', type=str, help='left disparity map')


def hisEqulColor(img):
    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    channels=cv2.split(ycrcb)
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
    return img

def disparity_to_gray(disp):
    image = np.zeros((disp.shape[0], disp.shape[1], 3), dtype=np.uint8)
    is_occluded = disp < 0
    image[:] = np.where(is_occluded, 0, 255 * disp / disp.max())[:, :, np.newaxis]
    image[is_occluded] = [255, 255, 0]
    return image

# You can modify the function interface as you like
def computeDisp(Il, Ir, search_depth, **kwargs):
    Il = hisEqulColor(Il)
    Ir = hisEqulColor(Ir)

    disp = graph.disparity(
            Il, Ir,
            search_depth=search_depth,
            **kwargs
        ).astype(np.float32)

   
    disp = cv2.boxFilter(disp, -1, (3, 3))
    disp = cv2.ximgproc.jointBilateralFilter(Il.astype(np.float32), disp, 11, 0.1, 3)

     # >>> Hole filling
    refinement.hole_filling(disp)

    disp = cv2.ximgproc.weightedMedianFilter(Il, disp, 11, cv2.ximgproc.WMF_EXP)


    return disp

def main():
    args = parser.parse_args()
    '''
    print(args.output)
    print('Compute disparity for %s' % args.input_left)
    img_left = cv2.imread(args.input_left)
    img_right = cv2.imread(args.input_right)
    tic = time.time()
    disp = computeDisp(img_left, img_right,search_depth=20, census_kernel_size=7)
    toc = time.time()
    cv2.imwrite(args.output.split('/')[-1],disparity_to_gray(disp))
    writePFM(args.output, disp)
    print('Elapsed time: %f sec.' % (toc - tic))
    '''
    import glob
    
    for t in types:
        filesL, filesR = (sorted(glob.glob('./data/Real/*L*.bmp')), sorted(glob.glob('./data/Real/*R*.bmp'))) \
        if t=='Real' else (sorted(glob.glob('./data/Synthetic/*L*.png')), sorted(glob.glob('./data/Synthetic/*R*.png')))

        #filesL, filesR = ['./data/cones/im0.png','./data/tsukuba/im0.png'], ['./data/cones/im1.png','./data/tsukuba/im1.png']
        for i, (fileL, fileR) in enumerate(zip(filesL, filesR )):
            print('Start %d-th image' %i)
            img_left = cv2.imread(fileL)
            img_right = cv2.imread(fileR)
            tic = time.time()
            disp = computeDisp(img_left, img_right, search_depth=15, occlusion_cost=-1, max_iterations=4,census_kernel_size=7, dissim_method='census')
            toc = time.time()
            cv2.imwrite('result_'+t+'_%d.png' %i ,disparity_to_gray(disp))
            writePFM('result_'+t+'_%d.pfm' %i, disp)
            print('Elapsed time: %f sec.' % (toc - tic))

if __name__ == '__main__':
    main()
