import tensorflow as tf
from PIL import Image
from utils import visualization_utils as vis_utils
import json
import os
import random

input_dir = "../../../mmdetection/landmine/train/"
all_img_names = []

with open(input_dir + 'coco_annotation.json') as json_file:
    annotations = json.load(json_file)

for img_name in os.scandir(input_dir):
    if img_name.name.endswith("tif"):
        all_img_names.append(img_name.name)

for img_name in all_img_names:
    # here get all the annotations that correspond to one image idx
    for img_entry in annotations['images']:
        if img_entry['file_name'] == img_name:
            img_id = img_entry['id']
    annots = [an for an in annotations["annotations"] if an['image_id'] == img_id]
    
    img_path = os.path.join(input_dir, img_name)
    img = Image.open(img_path)
    for annot in annots:
        tlx, tly, w, h = annot['bbox']
        cat = annot['category_id']
        caption = annotations['categories'][cat]['name']
        vis_utils.draw_bounding_box_on_image(img, tly, tlx, tly+h, tlx+w,
                display_str_list = [caption],
                use_normalized_coordinates=False)

    img.save(os.path.join(input_dir, "Viz/", img_name))
