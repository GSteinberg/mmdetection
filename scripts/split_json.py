# takes json file in home directory and outputs json with annotations only for the files in the
# directory listed as the parameter

import os
import sys
import json
import pdb

target_dir = sys.argv[1]
dir_w_ann = "../SplitData/COCO_no_slivers/"
# contents of target dir
dir_cont = os.listdir(target_dir)

# all annotations
with open(dir_w_ann + 'coco_annotation.json') as json_file:
    annot = json.load(json_file)

# handling images section
idx = 0 
while idx < len(annot['images']):
    img = annot['images'][idx]
    if img['file_name'] not in dir_cont:
        annot['images'].pop(idx)
    else:
        idx+=1

# handling annotations section
img_ids = [i['id'] for i in annot['images']]
idx = 0
while idx < len(annot['annotations']):
    ann = annot['annotations'][idx]
    if ann['image_id'] not in img_ids:
        annot['annotations'].pop(idx)
    else:
        idx+=1

# handling categories section
cat_ids = set([i['category_id'] for i in annot['annotations']])
idx = 0
while idx < len(annot['categories']):
    cat = annot['categories'][idx]
    if cat['id'] not in cat_ids:
        annot['categories'].pop(idx)
    else:
        idx+=1

# printing
with open(target_dir + 'coco_annotation.json', 'w') as outfile:
    json.dump(annot, outfile)
