import albumentations as A
import cv2
import argparse
import os
import json

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Augment images')
    parser.add_argument('--input_dirs', dest='input_dirs',
                        help='directories to take input imgs and anns to split',
                        nargs="+")
    parser.add_argument('--output_dir', dest='output_dir',
                        help='directory to save augmented imgs and ann',
                        type=str)
    args = parser.parse_args()
    return args


def get_annot_for_img(img_name, annot):
    # get image id associated with name
    img_id = -1
    for img in annot['images']:
        if 'file_name' == img_name:
            img_id = img['id']

    bboxes = []
    for ann in annot['annotations']:
        if ann['image_id'] == img_id:
            # get category name
            cat_id = ann['category_id']
            cat = annot['categories'][cat_id]

            # append entry
            bboxes.append(ann['bbox'] + cat)

    return bboxes


def augment(input_dirs, output_dir):
    # iterate through every input img directory
    for dir_num, input_dir in enumerate(input_dirs):
        # load coco annotations
        ann_name = os.path.join(input_dir, "coco_annotation.json")
        with open(ann_name) as json_file:
            annot = json.load(json_file)

        # iterate through every image in input_dirs
        for image in os.scandir(input_dir):
            # only check images with correct extension
            if not image.name.endswith(".tif"):
                print('{} not being parsed - does not have .tif extension'.format(image.name))
                continue

            # load image
            img = cv2.imread(image.path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # get corresponding annotation
            bboxes = get_annot_for_img(image.name, annot)

            # create transform object
            transform = A.Compose([
                A.RandomCrop(width=450, height=450),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2)
            ],
                bbox_params=A.BboxParams(
                    format='coco',
                    min_area=600,
                    min_visibility=0.4,
                    label_fields=['class_labels']
                )
            )

            # do actual transformation
            transformed = transform(image=img, bboxes=bboxes)
            transformed_img = transformed['image']
            transformed_bboxes = transformed['bboxes']


if __name__ == '__main__':

    args = parse_args()
    
    print("Called with args:")
    print(args)

    augment(args.input_dirs, args.output_dir)
