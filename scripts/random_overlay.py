### Overlay isolated objects onto orthos ###
# Pick objects randomly - set num of pfm and ksf per ortho
# Place objects in orthos randomly
# Create corresponding annotations

from os import listdir, scandir
from os.path import isfile, join
import sys
import json

OBJ_DIR = sys.argv[1]
ORTHO_DIR = sys.argv[2]
NUM_PFM = 26
NUM_KSF = 6

pfm_files = [f for f in listdir(OBJ_DIR) if isfile(join(OBJ_DIR, f)) and 'PFM' in f]
ksf_files = [f for f in listdir(OBJ_DIR) if isfile(join(OBJ_DIR, f)) and 'KSF' in f]

for ortho in scandir(ORTHO_DIR):
    
