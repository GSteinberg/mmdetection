import cv2
import numpy as np
import os
import math

os.chdir("data")
MID_BOX = 6

for curr_dir, subdirs, files in os.walk('./'):
    if curr_dir == './': continue
    
    # only iterate through images in subdirectories
    for img_name in files:
        print("Lassoing " + img_name)
        img = cv2.imread(os.path.join(curr_dir, img_name))      # load img
        
        height, width = img.shape[:2]
        rect = (50, 50, width-100, height-100)                # rectangle tightly containing the object
        
        # create a mask 
        mask = np.zeros((height,width),np.uint8)     # where is bg, fg

        # used by algorithm internally
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        
        # mode: whether we are drawing rectangle or final touchup strokes
        mode = cv2.GC_INIT_WITH_RECT
        # the # of iterations the algorithm should run
        iterations = 10

        # run
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterations, mode)
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        img = img * mask2[:,:,np.newaxis]       # multiple by original image
        
        # black background -> transparent
        tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
        b, g, r = cv2.split(img)
        rgba = [b, g, r, alpha]
        img = cv2.merge(rgba, 4)
        
        # save image
        new_name = '{}/{}_Lasso.tif'.format(curr_dir, img_name.split('.')[0])
        cv2.imwrite(new_name, img)
