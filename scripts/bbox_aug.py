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


def augment(input_dir, output_dir):
    # load coco annotations
    ann_name = os.path.join(input_dir, "coco_annotation.json")
    with open(ann_name) as json_file:
        annot = json.load(json_file)

    # for new images
    new_annot = {'images':[], 'annotations':[], 'categories':[]}
    img_id = -1

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
                min_visibility=0.4
            )
        )

        # do actual transformation
        transformed = transform(image=img, bboxes=bboxes)
        transformed_img = transformed['image']
        transformed_bboxes = transformed['bboxes']

        # image output
        output_img_name = os.path.join(output_dir, "Aug_" + image.name)
        cv2.imwrite(output_img_name, transformed_img)

        # reconstruct new coco ann
        new_annot['images']transformed_bboxes
        output_img_name

    # annotation output
    output_ann_name = os.path.join(output_dir, "coco_annotation.json")
    with open(output_ann_name, 'w') as outfile:
        json.dump(new_annot, outfile)


if __name__ == '__main__':

    args = parse_args()
    
    print("Called with args:")
    print(args)

    augment(args.input_dir, args.output_dir)
