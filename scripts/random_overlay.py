### Overlay isolated objects onto orthos ###
# Pick objects randomly - set num of pfm and ksf per ortho
# Place objects in orthos randomly
# Create corresponding annotations

from os import listdir, scandir
from os.path import isfile, join
import sys
import random

import xml.etree.ElementTree as ET
from PIL import Image


OBJ_DIR = sys.argv[1]
ORTHO_DIR = sys.argv[2]
OUT_DIR = sys.argv[3]
NUM_PFM = 26
NUM_KSF = 6


class BoundingBox:
    def __init__(self, name, trunc, diff, xmin, ymin, xmax, ymax):
        self.name = name
        self.truncated = trunc
        self.difficult = diff
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


# add new object to current xml tree
def new_object(umbrella_elemt, box_name, box_xmin, box_ymin, box_xmax, box_ymax):
    # create object xml tree
    obj = ET.SubElement(umbrella_elemt, 'object')
    name = ET.SubElement(obj, 'name')
    truncated = ET.SubElement(obj, 'truncated')
    difficult = ET.SubElement(obj, 'difficult')
    # create bndbox xml tree
    bndbox = ET.SubElement(obj, 'bndbox')
    xmin = ET.SubElement(bndbox, 'xmin')
    ymin = ET.SubElement(bndbox, 'ymin')
    xmax = ET.SubElement(bndbox, 'xmax')
    ymax = ET.SubElement(bndbox, 'ymax')

    # fill obj tree
    name.text = box_name
    truncated.text = '0'
    difficult.text = '0'
    # fill bndbox tree
    xmin.text = str(box_xmin)
    ymin.text = str(box_ymin)
    xmax.text = str(box_xmax)
    ymax.text = str(box_ymax)


if __name__ == '__main__':
    # box inside which mines should go
    orth_tl = (2803, 1329)
    orth_tr = (8900, 4989)
    orth_bl = (1088, 4325)
    orth_br = (7253, 7937)

    pfm_imgs = [OBJ_DIR+f for f in listdir(OBJ_DIR) if isfile(join(OBJ_DIR, f)) and 'PFM' in f]
    ksf_imgs = [OBJ_DIR+f for f in listdir(OBJ_DIR) if isfile(join(OBJ_DIR, f)) and 'KSF' in f]

    for ortho in scandir(ORTHO_DIR):
        # open orthophoto
        bg = Image.open(ortho.path)
        # bg_wid, bg_hei = bg.size
        bg_wid, bg_hei = 6788, 5064
        bg_x, bg_y = 1136, 1378

        # Create basic xml structure for writing
        ann = ET.Element('annotation')
        filename = ET.SubElement(ann, 'filename')
        size = ET.SubElement(ann, 'size')
        width = ET.SubElement(size, 'width')
        height = ET.SubElement(size, 'height')
        depth = ET.SubElement(size, 'depth')
        
        # size of cropped img
        img_width, img_height = bg.size
        # write size data to xml
        width.text = str(img_width)
        height.text = str(img_height)
        depth.text = "3"

        # select correct amounts of pfm and ksf
        objects = random.sample(pfm_imgs, NUM_PFM) + random.sample(ksf_imgs, NUM_KSF)

        # loop through each pfm-1
        for obj in objects:
            print('Overlaying {} onto {}'.format(obj.split('/')[-1], ortho.name))
            
            fg = Image.open(obj)
            fg_wid, fg_hei = fg.size

            # get x and y coordinates for top left pixel of pfm
            x_lim, y_lim = bg_wid-fg_wid, bg_hei-fg_hei 
            tl_x, tl_y = random.randint(bg_x, x_lim), random.randint(bg_y, y_lim)

            # overlay
            bg.paste(fg, (tl_x,tl_y), mask=fg.convert('RGBA'))

            # add new object to annotation
            xmin = tl_x
            ymin = tl_y
            xmax = tl_x + bg_wid
            ymax = tl_y + bg_hei
            obj_name = 'pfm-1' if 'PFM' in obj else'ksf-casing'
            new_object(ann, obj_name, xmin, ymin, xmax, ymax)

        # save image
        bg.save( join(OUT_DIR, ortho.name) )

        # convert xml tree to string
        root = ET.tostring(ann, encoding='unicode')
        ann.clear()
        
        ortho_ann = ortho.name.split('.')[0] + '.xml'
        xmlfile = open(ortho_ann, 'w')
        xmlfile.write(root)                     # write xml
