import tensorflow as tf
import sys
from PIL import Image
from utils import visualization_utils as vis_utils
import os
import random

ortho_dir = "../../../OrthoData/rubbOrth1/images/"
coord_file = sys.argv[1]
output_dir = sys.argv[2]
data = {}

with open(coord_file) as f:
    for line in f:
        entry = line.split()
        ortho = entry.pop(0)
        coords = [int(float(i)) for i in entry]

        if ortho not in data.keys():
            data[ortho] = []
        data[ortho].append(coords)
        

for ortho in data.keys():
    img = Image.open(os.path.join(ortho_dir, ortho))
    for elmt in data[ortho]:
        tlx, tly, brx, bry = elmt[:-1]
        caption = str(elmt[-1])
        vis_utils.draw_bounding_box_on_image(img, tly, tlx, brx, bry,
                display_str_list = [caption],
                use_normalized_coordinates=False)

    img.save(os.path.join(output_dir, "Res_" + ortho))
