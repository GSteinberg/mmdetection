import albumentations as A
import cv2
import argparse
import os
import json
from progress.bar import IncrementalBar

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Augment images')
    parser.add_argument('--input_dir', dest='input_dir',
                        help='directory to take input imgs and anns to split',
                        type=str)
    parser.add_argument('--output_dir', dest='output_dir',
                        help='directory to save augmented imgs and ann',
                        type=str)
    args = parser.parse_args()
    return args


def get_annot_for_img(img_name, annot):
    # get image id associated with name
    img_id = -1
    for img in annot['images']:
        if img['file_name'] == img_name:
            img_id = img['id']

    bboxes = []
    cats = []
    for ann in annot['annotations']:
        if ann['image_id'] == img_id:
            # get category name
            cat_id = ann['category_id']
            cat = annot['categories'][cat_id]

            # append entry
            bboxes.append(ann['bbox'])
            cats.append(cat)

    return bboxes, cats


def albument():
    # decide transformations
    transform_lst = [
        A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2)],
            bbox_params=A.BboxParams(
                format='coco',
                min_area=600,
                min_visibility=0.4,
                label_fields=['class_categories']
            )
        )
    ]

    return transform_lst


def augment(input_dir, output_dir):
    # load coco annotations
    ann_name = os.path.join(input_dir, "coco_annotation.json")
    with open(ann_name) as json_file:
        annot = json.load(json_file)

    # for new images
    new_annot = {'images':[], 'annotations':[], 'categories':[]}
    img_id = 0
    box_id = 0

    # create transform objects
    transform_lst = albument()

    # for output viz
    bar = IncrementalBar("Transforming images in " + input_dir, max=len(os.listdir(input_dir))*len(transform_lst))

    # iterate through every image in input_dirs
    for image in os.scandir(input_dir):
        # only check images with correct extension
        if not image.name.endswith(".tif"):
            print('\n{} not being parsed - does not have .tif extension'.format(image.name))
            bar.next()
            continue

        # load image
        img = cv2.imread(image.path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # get corresponding annotation
        bboxes, cats = get_annot_for_img(image.name, annot)

        # do actual transformations
        for tr_idx, tr in enumerate(transform_lst):
            transformed = tr(image=img, bboxes=bboxes, class_categories=cats)
            transformed_img = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_cats = transformed['class_categories']

            # image output
            output_img_name = "Aug{:02d}_{}".format(tr_idx, image.name)
            cv2.imwrite(os.path.join(output_dir, output_img_name), transformed_img)

            # reconstruct new coco ann
            # image entry
            img_height, img_width = transformed_img.shape[:2]
            new_annot['images'].append({
                'id': img_id,
                'file_name': output_img_name,
                'height': img_height,
                'width': img_width
            })

            # annotation entry
            for i, box in enumerate(transformed_bboxes):
                box = [int(coord) for coord in box]
                x1, y1, h, w = box
                x2 = x1 + w
                y2 = y1 + h
                area = h * w
                seg = [[x1,y1 , x2,y1 , x2,y2 , x1,y2]]

                new_annot['annotations'].append({
                    'image_id': img_id,
                    'id': box_id,
                    'category_id': transformed_cats[i]['id'],
                    'bbox': box,
                    'area': area,
                    'segmentation': seg,
                    'iscrowd': 0
                })
                box_id+=1

            # categories entry
            for cat in transformed_cats:
                if cat not in new_annot['categories']:
                    new_annot['categories'].append(cat)

            img_id+=1

        bar.next()

    bar.finish()

    # annotation output
    output_ann_name = os.path.join(output_dir, "coco_annotation.json")
    with open(output_ann_name, 'w') as outfile:
        json.dump(new_annot, outfile)


if __name__ == '__main__':

    args = parse_args()
    
    print("Called with args:")
    print(args)

    augment(args.input_dir, args.output_dir)
