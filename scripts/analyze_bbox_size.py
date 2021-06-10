import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import os
import pdb
import xml.etree.ElementTree as ET


class BoundingBox:
    def __init__(self, cat_name, cat_id, xmin, ymin, xmax, ymax):
        self.cat_name = cat_name
        self.cat_id = cat_id
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def read_xml(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bndboxes = []
    categories = []
    for boxes in root.iter('object'):
        name = boxes.find("name").text
        if name not in categories:
            categories.append(name)

    categories.sort()
        
    for boxes in root.iter('object'):
        name = boxes.find("name").text
        cat_id = categories.index(name)

        for box in boxes.findall("bndbox"):
            xmin = int(box.find("xmin").text)
            ymin = int(box.find("ymin").text)
            xmax = int(box.find("xmax").text)
            ymax = int(box.find("ymax").text)
        
        bb = BoundingBox(name, cat_id, xmin, ymin, xmax, ymax)
        bndboxes.append(bb)

    return bndboxes, categories


if __name__ == '__main__':
    type_ann = sys.argv[1]

    if type_ann.lower() == "json":
        # read annotation
        input_file = sys.argv[2]
        with open(input_file) as json_file:
            annot = json.load(json_file)

        cat_names = [cat['name'] for cat in annot['categories']]
        data = [[] for _ in cat_names]

        for ann in annot['annotations']:
            cat_id = ann['category_id']
            area = ann['area']
            data[cat_id].append(area)
    elif type_ann.lower() == "xml":
        ann_dirs = sys.argv[2:]
        data = {}
        num_cats = 0
        name_cats = ""

        # get anns from folders
        for ann_dir in ann_dirs:
            for ann in os.scandir(ann_dir):
                bndboxes, categories = read_xml(ann.path)
                if not categories: continue
                num_cats = len(categories)
                name_cats = categories
                data[ann.name] = [[] for _ in categories]

                # get area of each obj
                for bbox in bndboxes:
                    width = bbox.xmax - bbox.xmin
                    height = bbox.ymax - bbox.ymin
                    area = width * height

                    data[ann.name][bbox.cat_id].append(area)
    
        # compile total at end
        total = [[] for _ in range(num_cats)]
        for img_name, area_arr in data.items():
            for idx, arr in enumerate(area_arr):
                total[idx].extend(arr)
        # add total key value pair
        data['total'] = total
    
    # make plots for each category
    for idx, cat in enumerate(name_cats):
        fig, ax = plt.subplots(figsize=(18,18))
        ax.set_title('Analysis of Bounding Box Areas for {} in All Images'.format(cat))
        curr_area_arr = [area_arr[idx] for _, area_arr in data.items()]
        img_names = [img for img in data]
        ax.boxplot(curr_area_arr, labels=img_names)
        plt.xticks(rotation=90)
        ax.set_xlabel('image names')
        ax.set_ylabel('area (px)')

        plt.savefig("Size_analysis_{}.png".format(cat.lower()))
