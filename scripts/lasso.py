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

        # run
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, mode)
        mask = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        img = img * mask2[:,:,np.newaxis]       # multiple by original image
        
        # middle box fix
        # xmid = width / 2
        # ymid = height / 2

        # xmin = math.floor(xmid - MID_BOX)
        # xmax = math.floor(xmid + (MID_BOX-1) + 1*(xmid%1 > 0))
        # ymin = math.floor(ymid - MID_BOX)
        # ymax = math.floor(ymid + (MID_BOX-1) + 1*(ymid%1 > 0))

        # if not any(any(row[xmin:xmax+1]) for row in mask[ymin:ymax+1]):
        #     # make center box
        #     for row in range(height):
        #         for col in range(width):
        #             if xmin <= col <= xmax and ymin <= row <= ymax: mask[row][col] = 1

        # # mode: whether we are drawing rectangle or final touchup strokes
        # mode = cv2.GC_INIT_WITH_MASK
        # mask, bgdModel, fgdModel = cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, mode)

        # 0 and 2-pixels put to 0 (ie bg)
        # 1 and 3-pixels put to 1 (ie fg)
        # mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')

        # save image
        new_name = '{}/{}_Lasso.tif'.format(curr_dir, img_name.split('.')[0])
        cv2.imwrite(new_name, img)
