from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import csv
import mmcv
import numpy as np
import os
import json
import utm
import torch
import torch.distributed as dist
from mmcv.runner import BaseModule, auto_fp16

from mmdet.core.visualization import imshow_det_bboxes


class BaseDetector(BaseModule, metaclass=ABCMeta):
    """Base class for detectors."""

    def __init__(self, init_cfg=None):
        super(BaseDetector, self).__init__(init_cfg)
        self.fp16_enabled = False

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    # TODO: these properties need to be carefully handled
    # for both single stage & two stage detectors
    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_bbox)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_mask)
                or (hasattr(self, 'mask_head') and self.mask_head is not None))

    @abstractmethod
    def extract_feat(self, imgs):
        """Extract features from images."""
        pass

    def extract_feats(self, imgs):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]

    def forward_train(self, imgs, img_metas, **kwargs):
        """
        Args:
            img (list[Tensor]): List of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        batch_input_shape = tuple(imgs[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

    async def async_simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def simple_test(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

    async def aforward_test(self, *, img, img_metas, **kwargs):
        for var, name in [(img, 'img'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(img)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(img)}) '
                             f'!= num of image metas ({len(img_metas)})')
        # TODO: remove the restriction of samples_per_gpu == 1 when prepared
        samples_per_gpu = img[0].size(0)
        assert samples_per_gpu == 1

        if num_augs == 1:
            return await self.async_simple_test(img[0], img_metas[0], **kwargs)
        else:
            raise NotImplementedError

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        if isinstance(img, str):
            img_name = img.split("/")[-1]
        else:
            img_name = out_file.split("/")[-1]
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        coords = {}
        ortho_coords = {}
        for idx, bbox in enumerate(bboxes):
            # if score is greater than score threshold
            if bbox[4] > score_thr:
                # row and col of image in respective orthophoto (img_ortho)
                split_img_name = img_name.split("_Split")
                img_row, img_col = int(split_img_name[1][:3]), int(split_img_name[1][3:6])
                img_ortho = split_img_name[0]
                full_img_ortho_name = img_ortho + ".tif"
            
                # 2 elemt list of pixel of center of landmine
                # structure: [average(xmin, xmax), average(ymin, ymax)]
                cropped_px = [ int( np.mean([int(bbox[0]),int(bbox[2])]) ),
                               int( np.mean([int(bbox[1]),int(bbox[3])]) ) ]
                score = float(bbox[4])

                # getting cropped img size and stride
                with open('../SplitData/COCO/meta_dict.json') as json_file:
                    meta_sizes = json.load(json_file)
                size_minus_stride = meta_sizes[full_img_ortho_name][0] - meta_sizes[full_img_ortho_name][1]

                # print for drawing on ortho
                xmin_p = bbox[0] + (img_col*size_minus_stride)
                ymin_p = bbox[1] + (img_row*size_minus_stride)
                xmax_p = bbox[2] + (img_col*size_minus_stride)
                ymax_p = bbox[3] + (img_row*size_minus_stride)
                # print("{}.tif {} {} {} {} {}".format(img_ortho, xmin_p, ymin_p, xmax_p, ymax_p, labels[idx]))

                # converting to orthophoto scale
                ortho_x, ortho_y = cropped_px[0] + (img_col*size_minus_stride), \
                                   cropped_px[1] + (img_row*size_minus_stride)

                # fetch respective ortho metdata
                # structure: metadata[0]    == x-pixel res
                #            metadata[1:3]  == rotational components
                #            metadata[3]    == y-pixel res
                #            metadata[4]    == Easting of upper left pixel
                #            metadata[5]    == Northing of upper left pixel
                ortho_dir = os.path.join("../OrthoData/metadata", \
                            img_ortho + ".tfw")
                f = open(ortho_dir, "r")
                metadata = f.read().split("\n")[:-1]
                f.close()

                x_res, y_res, easting, northing = \
                        float(metadata[0]), float(metadata[3]), \
                        float(metadata[4]), float(metadata[5])

                if img_ortho not in coords.keys():
                    coords[img_ortho] = []
                coords[img_ortho].append([labels[idx], score,
                        easting + (ortho_x*x_res), northing + (ortho_y*y_res)])

                # output for orthophoto level eval
                if img_ortho not in ortho_coords.keys():
                    ortho_coords[img_ortho] = []
                ortho_coords[img_ortho].append([labels[idx], score, ortho_x, ortho_y])

        if coords:
            # ortho this photo belongs to
            img_name = [*coords][0]

            # convert utm to lat long
            for pnt in range(len(coords[img_name])):
                lat_long = utm.to_latlon(coords[img_name][pnt][2], coords[img_name][pnt][3], 18, 'T')
                coords[img_name][pnt].extend(lat_long)

            # coords for each ortho
            indv_ortho_file = 'faster_rcnn_r101_fpn_1x_coco_results/' + img_name + '_coords.csv'
            with open(indv_ortho_file, 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # if file is empty, write heading then append, else just append
                if os.stat(indv_ortho_file).st_size == 0:
                    writer.writerow(["Object", "Score", "Easting", "Northing", "Latitude", "Longitude"])
                for c in coords[img_name]:
                    writer.writerow(c[:])

            # all coords from all orthos
            all_coords_file = 'faster_rcnn_r101_fpn_1x_coco_results/all_coords.csv'
            with open(all_coords_file, 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # if file is empty, write heading then append, else just append
                if os.stat(all_coords_file).st_size == 0:
                    writer.writerow(["Photo", "Object", "Score", "Easting", "Northing", "Latitude", "Longitude"])
                for c in coords[img_name]:
                    writer.writerow([img_name] + c[:])

        # draw bounding boxes
        # img = imshow_det_bboxes(
        #     img,
        #     bboxes,
        #     labels,
        #     segms,
        #     class_names=self.CLASSES,
        #     score_thr=score_thr,
        #     bbox_color=bbox_color,
        #     text_color=text_color,
        #     mask_color=mask_color,
        #     thickness=thickness,
        #     font_size=font_size,
        #     win_name=win_name,
        #     show=show,
        #     wait_time=wait_time,
        #     out_file=out_file)

        if not (show or out_file):
            pass
            # return img
        else:
            return ortho_coords
