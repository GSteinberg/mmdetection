import os.path as osp
import pickle
import shutil
import tempfile
import time
import math
import xml.etree.ElementTree as ET

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    pd = {}
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                curr_pd = model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

                # add current preds to preds
                for ortho_name in curr_pd.keys():
                    if ortho_name not in pd.keys():
                        pd[ortho_name] = curr_pd[ortho_name]
                    else:
                        pd[ortho_name].extend(curr_pd[ortho_name])

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    
    # get ground truth for ortho level eval
    gt = {ortho_name: [] for ortho_name in pd.keys()}

    for ortho_name in gt.keys():
        if "Test" not in ortho_name:
            ortho_path = ortho_name.split("_")[0] + "/annotations/"
        else:
            ortho_path = "Rubble_Test/annotations/"
        tree = ET.parse("../OrthoData/" + ortho_path + ortho_name + ".xml")
        root = tree.getroot()

        for boxes in root.iter('object'):
            name = boxes.find("name").text
            for box in boxes.findall("bndbox"):
                xmin, ymin = int(box.find("xmin").text), int(box.find("ymin").text)
                xmax, ymax = int(box.find("xmax").text), int(box.find("ymax").text)
            
            entry = [name, int((xmin+xmax) / 2), int((ymin+ymax) / 2)]
            gt[ortho_name].append(entry)

    # prepare error dict - 2 classes
    classes = ["pfm-1", "ksf-casing"]
    raw_err = {ortho_name : [{"tp":0, "fp":0, "fn":0} for _ in range(len(classes))] for ortho_name in gt.keys()}
    tot_gt = {ortho_name : [0 for _ in range(len(classes))] for ortho_name in gt.keys()}

    # ortho level evaluation
    min_dist = 20
    for ortho in pd.keys():
        # for each predicted point in respective ortho
        for pd_entry in pd[ortho]:
            match = False       # prevent duplicate matches

            # each gt point for same ortho
            for gt_entry in gt[ortho]:
                # if different object - go to next gt
                pd_cat = pd_entry[0]
                gt_cat = classes.index(gt_entry[0].lower())
                if pd_cat != gt_cat: continue

                pd_coord = pd_entry[2:]
                gt_coord = gt_entry[1:]
                dist = math.sqrt(sum([(a - b) ** 2 for a, b in zip(pd_coord,gt_coord)]))

                # TP: pred px matches ground truth px only once
                if dist < min_dist and not match: 
                    raw_err[ortho][pd_cat]['tp'] += 1
                    match = True
            
            # FP: no truth box to match pred box
            if not match: 
                raw_err[ortho][pd_cat]['fp'] += 1
    
        # total ground truths for each cat
        for gt_entry in gt[ortho]:
            class_name = gt_entry[0].lower()
            if class_name not in classes:
                print("class is not in given classes")
                exit()

            class_i = classes.index(class_name)
            tot_gt[ortho][class_i] += 1

        # FN: if # accurately predicted boxes < total # ground truths
        for cat_i in range(len(classes)):
            if raw_err[ortho][cat_i]['tp'] < tot_gt[ortho][cat_i]:
                raw_err[ortho][cat_i]['fn'] += tot_gt[ortho][cat_i] - raw_err[ortho][cat_i]['tp']

    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
