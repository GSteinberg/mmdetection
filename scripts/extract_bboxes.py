import glob
import os
import xmltodict
import json
import pprint

from PIL import Image

# Look for XML files and parses then as if they were Pascal VOC Files
def process():
    # Finds all XML files on data/ and append to list
    pv_cont = []
    os.chdir("data")

    print("Found {} files in data directory!".format( str(len(glob.glob("*.xml"))) ))
    for ann in glob.glob("*.xml"):
        f_handle = open(ann, 'r')
        print("Parsing file '{}'...".format(ann))
        pv_cont.append(xmltodict.parse(f_handle.read()))

    # Process each file individually
    for pv_ann in pv_cont:
        image_file = pv_ann['annotation']['filename']
        # If there's a corresponding file in the folder,
        # process the images and save to output folder
        if os.path.isfile(image_file):
            extractDataset(pv_ann['annotation'])
        else:
            print("Image file '{}' not found, skipping file...".format(image_file))


# Extract image samples and save to output dir
def extractDataset(dataset):
    if 'object' in dataset.keys():
        print("Found {} objects on image '{}'...".format(len(dataset['object']), dataset['filename']))
    else:
        print("Found 0 objects on image '{}'...".format(dataset['filename']))
        return

    # Open image and get ready to process
    img = Image.open(dataset['filename'])

    # Create output directory
    save_dir = dataset['filename'].split('.')[0]
    try:
        os.mkdir(save_dir)
    except:
        pass
    
    # Image name preamble
    sample_preamble = '{}/{}'.format(save_dir, dataset['filename'].split('.')[0])
    
    # Image counter
    counter = {}
    # Run through each item and save cut image to output folder
    for item in dataset['object']:
        # Convert str to integers
        bndbox = dict([(a, int(b)) for (a, b) in item['bndbox'].items()])
        
        # Crop image
        xmin = bndbox['xmin'] - 50
        ymin = bndbox['ymin'] - 50
        xmax = bndbox['xmax'] + 50
        ymax = bndbox['ymax'] + 50
        im = img.crop((xmin, ymin, xmax, ymax))

        cat = item['name']
        if cat in counter.keys():
            counter[cat] += 1
        else:
            counter[cat] = 1

        # Save
        im.save('{}_{}_{:02d}.tif'.format(sample_preamble, cat, counter[cat]-1))

if __name__ == '__main__':
    process()
