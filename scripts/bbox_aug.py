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


def augment(input_dirs, output_dir):
    # iterate through every input img directory
    for dir_num, input_dir in enumerate(input_dirs):
        # load coco annotations
        ann_name = os.path.join(input_dir, "coco_annotation.json")
        with open(ann_name) as json_file:
            annot = json.load(json_file)

        # iterate through every image in input_dirs
        for image_name in os.scandir(input_dir):
            # only check images with correct extension
            if not image.name.endswith(".tif"):
                print('{} not being parsed - does not have .tif extension'.format(image.name))
                continue

            img = cv2.imread(image.path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            transform = A.Compose([
                A.RandomCrop(width=450, height=450),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
            ],  bbox_params=A.BboxParams(
                    format='coco',
                    min_area=600,
                    min_visibility=0.4,
                    label_fields=['class_labels']
                )
            )


if __name__ == '__main__':

    args = parse_args()
    
    print("Called with args:")
    print(args)

    augment(args.input_dirs, args.output_dir)
