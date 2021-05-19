import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict
import json
import math
import csv
import os
import utm
import xml.etree.ElementTree as ET

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class CocoDataset(CustomDataset):

    # CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    #            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    #            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    #            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    #            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    #            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    #            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    #            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    #            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    #            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    #            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    #            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    #            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    #            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    CLASSES = ('pfm-1', 'ksf-casing')

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            # comment out the below line to not train on background images
            # self.filter_empty_gt = False
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def get_test_img_names(self):
        # get image names to coordinate to img ids
        with open('landmine/test/coco_annotation.json') as outfile:
            imgNames = json.load(outfile)
        return imgNames

    def calc_tp_fp_fn(self, cat_ids, cntr_gts, cntr_dts, min_dist=8.5):
        num_classes = len(cat_ids)
        raw_err = [{"tp":0, "fp":0, "fn":0} for _ in range(num_classes)]
        for cat in cat_ids:
            for dt in cntr_dts[cat]:
                match = False       # prevent duplicate matches

                # for ground truth box of category cat for same image
                for gt in cntr_gts[cat]:
                    # if gt and dt are not in the same image, skip this gt
                    if gt[0] != dt[0]: continue

                    # in same image - get center points
                    dt_cnt = dt[-2:]
                    gt_cnt = gt[-2:]
                    dist = math.sqrt(sum([(a - b) ** 2 for a, b in zip(dt_cnt,gt_cnt)]))
                    # TP: pred px matches a ground truth px
                    if dist < min_dist:
                        if not gt[1]:
                            raw_err[cat]['tp'] += 1
                            gt[1] = True
                        match = True
                        break

                # FP: no truth box to match pred box
                if not match:
                    raw_err[cat]['fp'] += 1

            # FN: if # accurately predicted boxes < total # ground truths
            if raw_err[cat]['tp'] < len(cntr_gts[cat]):
                raw_err[cat]['fn'] += len(cntr_gts[cat]) - raw_err[cat]['tp']

        # add totals
        raw_total = {'tp':0, 'fp':0, 'fn':0}
        for key in raw_total.keys():
            raw_total[key] = sum(raw_err[c][key] for c in range(num_classes))
        raw_err.append(raw_total)

        return raw_err

    def calc_prec_reca_f1(self, cat_ids, raw_err):
        num_classes = len(cat_ids)
        rel_err = [{"prec":0, "recall":0, "f1":0} for _ in range(num_classes)]
        for c in range(num_classes):
            if raw_err[c]['tp'] == 0:
                rel_err[c]['prec'] = 0
                rel_err[c]['recall'] = 0
                rel_err[c]['f1'] = 0
                continue

            # precision - tp/(tp+fp)
            rel_err[c]['prec'] = raw_err[c]['tp'] / (raw_err[c]['tp']+raw_err[c]['fp'])
            # recall - tp/(tp+fn)
            rel_err[c]['recall'] = raw_err[c]['tp'] / (raw_err[c]['tp']+raw_err[c]['fn'])
            # f1 - 2*[(prec*rec)/(prec+rec)]
            rel_err[c]['f1'] = 2 * \
                    ((rel_err[c]['prec'] * rel_err[c]['recall']) / (rel_err[c]['prec'] + rel_err[c]['recall']))

        # average prec, recall, f1
        rel_total = {"prec":0, "recall":0, "f1":0}
        for key in rel_total.keys():
            rel_total[key] = np.mean([rel_err[c][key] for c in range(num_classes)])

        # add totals
        rel_err.append(rel_total)

        return rel_err

    def print_err_rep(self, cat_ids, raw_err, rel_err, name):
        with open("faster_rcnn_r101_fpn_1x_coco_results/" + name, "w", newline='') as f:
            writer = csv.writer(f)

            writer.writerow(["--"] + cat_ids + ["total"])
            for key in raw_err[0].keys():
                writer.writerow([key] + [raw_err[i][key] for i in range(len(raw_err))])
            writer.writerow(['----'])
            for key in rel_err[0].keys():
                writer.writerow([key] + ["{:.4f}".format(rel_err[i][key]) for i in range(len(rel_err))])

    def conv_to_ortho_scale(self, ortho_name, col, row, crop_x, crop_y):
        # for cropped img size and stride
        with open('../SplitData/COCO/meta_dict.json') as json_file:
            meta_sizes = json.load(json_file)

        # getting cropped img size and stride
        size_minus_stride = meta_sizes[ortho_name][0] - meta_sizes[ortho_name][1]

        # converting to orthophoto scale
        ortho_x, ortho_y = crop_x + (col*size_minus_stride), crop_y + (row*size_minus_stride)

        return ortho_x, ortho_y

    def gen_irl_and_ortho_coords(self, cat_names, cntr_dts, min_dist=8.5):
        coords = {}
        ortho_coords = {}

        imgNames = self.get_test_img_names()

        for cat_id in range(len(cntr_dts)):
            for bbox in cntr_dts[cat_id]:
                # row and col of image in respective orthophoto (img_ortho)
                img_name = next(entry['file_name'] for entry in imgNames['images'] if entry['id'] == bbox[0])
                split_img_name = img_name.split("_Split")
                img_row, img_col = int(split_img_name[1][:3]), int(split_img_name[1][3:6])
                img_ortho = split_img_name[0]
                full_img_ortho_name = img_ortho + ".tif"

                # converting to orthophoto scale
                ortho_x, ortho_y = self.conv_to_ortho_scale(full_img_ortho_name, img_col, img_row, bbox[-2], bbox[-1])

                # throw out any duplicate boxes
                dup = False
                curr_pt = [ortho_x, ortho_y]

                if img_ortho in ortho_coords.keys():
                    for pt in ortho_coords[img_ortho]:
                        dist = math.sqrt(sum([(a - b) ** 2 for a, b in zip(pt[2:],curr_pt)]))
                        if dist < min_dist:
                            dup = True
                            break
                if dup: continue

                # fetch respective ortho metdata
                # structure: metadata[0]    == x-pixel res
                #            metadata[1:3]  == rotational components
                #            metadata[3]    == y-pixel res
                #            metadata[4]    == Easting of upper left pixel
                #            metadata[5]    == Northing of upper left pixel
                ortho_dir = os.path.join("../OrthoData/metadata", img_ortho + ".tfw")
                f = open(ortho_dir, "r")
                metadata = f.read().split("\n")[:-1]
                f.close()

                x_res, y_res, easting, northing = \
                        float(metadata[0]), float(metadata[3]), float(metadata[4]), float(metadata[5])

                score = bbox[1]

                if img_ortho not in coords.keys():
                    coords[img_ortho] = []
                coords[img_ortho].append([cat_names[cat_id], score,
                        easting + (ortho_x*x_res), northing + (ortho_y*y_res)])

                # output for orthophoto level eval
                if img_ortho not in ortho_coords.keys():
                    ortho_coords[img_ortho] = []
                ortho_coords[img_ortho].append([cat_names[cat_id], score, ortho_x, ortho_y])

        return coords, ortho_coords

    def print_irl_coords(self, coords):
        # convert utm to lat long
        for img_name in coords.keys():
            for pnt in range(len(coords[img_name])):
                lat_long = utm.to_latlon(coords[img_name][pnt][2], coords[img_name][pnt][3], 18, 'T')
                coords[img_name][pnt].extend(lat_long)

        # coords for each ortho
        for img_name in coords.keys():
            indv_ortho_file = 'faster_rcnn_r101_fpn_1x_coco_results/' + img_name + '_coords.csv'
            with open(indv_ortho_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Object", "Score", "Easting", "Northing", "Latitude", "Longitude"])
                for c in coords[img_name]:
                    writer.writerow(c[:])

        # all coords from all orthos
        all_coords_file = 'faster_rcnn_r101_fpn_1x_coco_results/all_coords.csv'
        with open(all_coords_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Photo", "Object", "Score", "Easting", "Northing", "Latitude", "Longitude"])
            for img_name in coords:
                for c in coords[img_name]:
                    writer.writerow([img_name] + c[:])

    def remove_dup(self, cat_ids, cntr_dts):
        min_dist = 8.5
        num_classes = len(cat_ids)
        for cat in range(num_classes):
            pt1 = 0
            while pt1 < len(cntr_dts[cat]):
                dup = False
                pt2 = 0
                while pt2 < len(cntr_dts[cat]):
                    if cntr_dts[cat][pt1][0] != cntr_dts[cat][pt2][0]:
                        pt2+=1
                        continue

                    if pt1 == pt2:
                        pt2+=1
                        continue

                    coords1 = cntr_dts[cat][pt1][2:]
                    coords2 = cntr_dts[cat][pt2][2:]
                    dist = math.sqrt(sum([(a - b) ** 2 for a, b in zip(coords1, coords2)]))
                    if dist < min_dist:
                        cntr_dts[cat].pop(pt1)
                        dup = True
                        break
                    pt2+=1
                if not dup: pt1+=1
        return cntr_dts

    def ortho_lvl(self, cat_ids, cat_names, score_thr, cntr_dts):
        num_classes = len(cat_ids)

        ann = self.get_test_img_names()
        ortho_names = set(entry['file_name'].split("_Split")[0] for entry in ann['images'])
        ortho_names = sorted( list(ortho_names) )

        # get ground truth from orthophotos
        cntr_gts = [[] for _ in range(num_classes)]
        for ortho_name in ortho_names:
            # find corresponding orthophoto
            if "Test" not in ortho_name:
                ortho_path = ortho_name.split("_")[0] + "/annotations/"
            else:
                ortho_path = "Rubble_Test/annotations/"
            tree = ET.parse("../OrthoData/" + ortho_path + ortho_name + ".xml")
            root = tree.getroot()

            for boxes in root.iter('object'):
                cat = boxes.find("name").text.lower()
                cat_id = cat_names.index(cat)

                for box in boxes.findall("bndbox"):
                    xmin, ymin = int(box.find("xmin").text), int(box.find("ymin").text)
                    xmax, ymax = int(box.find("xmax").text), int(box.find("ymax").text)

                # [img_id, matched_flag, x_coord, y_coord]
                cntr_gts[cat_id].append([ortho_names.index(ortho_name), False, np.mean([xmin,xmax]), np.mean([ymin,ymax])])

        # modify cntr_dts img ids so they are at ortho level
        for cat in cntr_dts:
            for dt in cat:
                # convert img ids
                img_name = next(entry['file_name'] for entry in ann['images'] if entry['id'] == dt[0])
                split_img_name = img_name.split("_Split")
                img_row, img_col = int(split_img_name[1][:3]), int(split_img_name[1][3:6])
                img_ortho = split_img_name[0]

                ortho_id = ortho_names.index(img_ortho)
                dt[0] = ortho_id

                full_img_ortho_name = img_ortho + ".tif"

                # convert coords
                dt[2], dt[3] = self.conv_to_ortho_scale(full_img_ortho_name, img_col, img_row, dt[2], dt[3])

        # remove duplicates from cntr_dts
        cntr_dts = self.remove_dup(cat_ids, cntr_dts)

        # calculate raw err seperately for each orthophoto
        raw_err_sep = []
        for ortho_i in range(len(ortho_names)):
            curr_cntr_gts = []
            curr_cntr_dts = []
            for cat in range(len(cntr_gts)):
                curr_cntr_gts.append([entry for entry in cntr_gts[cat] if entry[0] == ortho_i])
                curr_cntr_dts.append([entry for entry in cntr_dts[cat] if entry[0] == ortho_i])

            raw_err_sep.append( self.calc_tp_fp_fn(cat_ids, curr_cntr_gts, curr_cntr_dts) )

        # calculate precision, recall, F1 for each orthophoto
        rel_err_sep = []
        for ortho_i in range(len(ortho_names)):
            rel_err_sep.append( self.calc_prec_reca_f1(cat_ids, raw_err_sep[ortho_i]) )

        # print error reports
        for ortho_i in range(len(ortho_names)):
            err_rep_name = "error_report_{}_{}.csv".format(str(score_thr)[2:], ortho_names[ortho_i])
            self.print_err_rep(cat_ids, raw_err_sep[ortho_i], rel_err_sep[ortho_i], err_rep_name)

        # print grand error report
        raw_err_tot = [{"tp":0, "fp":0, "fn":0} for _ in range(num_classes + 1)]
        rel_err_tot = [{"prec":0, "recall":0, "f1":0} for _ in range(num_classes + 1)]
        for ortho_i in range(len(ortho_names)):
            for cat in range(len(raw_err_sep[ortho_i])):
                for key in raw_err_sep[ortho_i][cat].keys():
                    raw_err_tot[cat][key] += raw_err_sep[ortho_i][cat][key]
                for key in rel_err_sep[ortho_i][cat].keys():
                    rel_err_tot[cat][key] += rel_err_sep[ortho_i][cat][key]

        for cat in range(len(rel_err_tot)):
            for key in rel_err_tot[cat].keys():
                rel_err_tot[cat][key] /= len(ortho_names)

        err_rep_name = "error_report_{}_Grand.csv".format(str(score_thr)[2:])
        self.print_err_rep(cat_ids, raw_err_tot, rel_err_tot, err_rep_name)

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                cocoDt = cocoGt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                ### =========================================================== ###

                imgIds = sorted(cocoGt.getImgIds())
                catIds = sorted(cocoGt.getCatIds())
                num_classes = len(catIds)
                gts=cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=imgIds, catIds=catIds))
                dts=cocoDt.loadAnns(cocoDt.getAnnIds(imgIds=imgIds, catIds=catIds))

                # names of cats
                cat_names = ['pfm-1', 'ksf-casing']

                # get center pxl for each ground truth box
                cntr_gts = [[] for _ in range(num_classes)]
                for gt in gts:
                    gt_catId = gt['category_id']
                    gt_imgId = gt['image_id']
                    gt_box = gt['segmentation'][0][:2] + gt['segmentation'][0][4:6]
                    # [img_id, x_coord, y_coord]
                    cntr_gts[gt_catId].append(
                            (gt_imgId, np.mean([gt_box[0], gt_box[2]]), np.mean([gt_box[1], gt_box[3]])))

                # True:  calc score for multiple thresholds to convert to a graph
                # False: calc score for one threshold
                AUC_chart = False
                for score_thr in np.arange(0, 1, 0.05):
                    # if no chart is wanted, set one threshold
                    if not AUC_chart:
                        score_thr = 0.6

                    score_thr = np.round(score_thr, 2)      # round for arange bug

                    # get center pxl for each detected box
                    cntr_dts = [[] for _ in range(num_classes)]
                    for dt in dts:
                        dt_catId = dt['category_id']
                        dt_imgId = dt['image_id']
                        dt_box = dt['segmentation'][0][:2] + dt['segmentation'][0][4:6]

                        # if score for that box > prediction threshold, add to cntr_dts
                        # [img_id, score, x_coord, y_coord]
                        if dt['score'] > score_thr:
                            cntr_dts[dt_catId].append([dt_imgId, dt['score'],
                                    np.mean([dt_box[0],dt_box[2]]), np.mean([dt_box[1],dt_box[3]])])

                    # calculate raw error
                    raw_err = self.calc_tp_fp_fn(catIds, cntr_gts, cntr_dts)
                    
                    # calculate precision, recall, F1 for each class and all classes
                    rel_err = self.calc_prec_reca_f1(catIds, raw_err)

                    # print error reports
                    err_rep_name = "error_report_{}.csv".format(str(score_thr)[2:])
                    self.print_err_rep(catIds, raw_err, rel_err, err_rep_name)

                    if not AUC_chart: break

                # True: output real world coords of detections
                output_coords = True
                if output_coords:
                    # generate real world coords and orthophoto coords
                    coords, ortho_coords = self.gen_irl_and_ortho_coords(cat_names, cntr_dts)

                    # OUTPUT REAL COORDS
                    self.print_irl_coords(coords)

                # True: do orthophoto-level evaluation
                ortho_lvl_eval = True
                if ortho_lvl_eval:
                    self.ortho_lvl(catIds, cat_names, score_thr, cntr_dts)

                ### =========================================================== ###

                # compute coco metrics
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
