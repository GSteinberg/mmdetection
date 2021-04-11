### Overlay isolated objects onto orthos ###
# Pick objects randomly - set num of pfm and ksf per ortho
# Place objects in orthos randomly
# Create corresponding annotations

from os import listdir, scandir
from os.path import isfile, join
import sys
import random

from PIL import Image

OBJ_DIR = sys.argv[1]
ORTHO_DIR = sys.argv[2]
OUT_DIR = sys.argv[3]
NUM_PFM = 26
NUM_KSF = 6

pfm_imgs = [OBJ_DIR+f for f in listdir(OBJ_DIR) if isfile(join(OBJ_DIR, f)) and 'PFM' in f]
ksf_imgs = [OBJ_DIR+f for f in listdir(OBJ_DIR) if isfile(join(OBJ_DIR, f)) and 'KSF' in f]

for ortho in scandir(ORTHO_DIR):
    # select correct amounts of pfm and ksf
    objects = random.sample(pfm_imgs, NUM_PFM) + random.sample(ksf_imgs, NUM_KSF)

    # open orthophoto
    bg = Image.open(ortho.path)
    bg_wid, bg_hei = bg.size

    # loop through each pfm-1
    for obj in objects:
        fg = Image.open(obj)
        fg_wid, fg_hei = fg.size

        # get x and y coordinates for top left pixel of pfm
        x_lim, y_lim = bg_wid-fg_wid, bg_hei-fg_hei
        tl_x, tl_y = random.randint(0,x_lim), random.randint(0,y_lim)

        # overlay
        bg.paste(fg, (tl_x,tl_y), mask=fg.convert('RGBA'))
        break

    import pdb;pdb.set_trace()
    bg.save( join(OUT_DIR, ortho.name) )
