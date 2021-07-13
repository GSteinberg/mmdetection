import json
import sys
from collections import Counter

input_file = sys.argv[1]
with open(input_file) as json_file:
    annot = json.load(json_file)

num_img_w_obj = len(set([ann['image_id'] for ann in annot['annotations']]))
num_imgs = len(annot['images'])

print("Number of images:  " + str(num_imgs))
print("Number of images with objects: " + str(num_img_w_obj))
print("Number of background images: " + str(num_imgs - num_img_w_obj))
print("Number of objects: " + str(len(annot['annotations'])))
print("---------------------------------")

objs = annot['categories']

cat_ids = Counter([str(el['category_id']) for el in annot['annotations']])
for o in objs:
    print("Number of obj {}: {}".format(o['name'], cat_ids[str(o['id'])]))
