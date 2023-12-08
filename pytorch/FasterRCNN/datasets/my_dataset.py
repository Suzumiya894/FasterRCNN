import numpy as np
import os
from pathlib import Path
import random
import xml.etree.ElementTree as ET
import json

from .training_sample import Box
from .training_sample import TrainingSample
from . import image
from pytorch.FasterRCNN.models import anchors

class Dataset:

    num_classes = 14
    class_index_to_name = {
        0:  "no_flaw",
        1:  "flaw_1",
        2:  "flaw_2",
        3:  "flaw_3",
        4:  "flaw_4",
        5:  "flaw_5",
        6:  "flaw_6",
        7:  "flaw_7",
        8:  "flaw_8",
        9:  "flaw_9",
        10: "flaw_10",
        11: "flaw_11",
        12: "flaw_12",
        13: "flaw_13",
    } 
    invalid_anna_id = {}
    invalid_anna_id['train'] = [0, 13, 51, 172, 202, 668, 725, 774, 791, 929, 954, 2946, 3966, 4551, 4737, 4885, 4928, 4930, 4941, 4983, 4999, 5020, 5401, 5420, 5432]
    invalid_anna_id['test'] = [15, 126, 128, 197, 204, 618, 619, 864, 1803]
    """
    dir = VOCdevkit/VOC2007
    split = trainval
    image_preprocessing_params = PreprocessingParams(channel_order=<ChannelOrder.BGR: 'BGR'>, scaling=1.0, means=[103.939, 116.779, 123.68], stds=[1, 1, 1])
    compute_feature_map_shape_fn = <bound method VGG16Backbone.compute_feature_map_shape of <pytorch.FasterRCNN.models.vgg16.VGG16Backbone object at 0x000001B6C0E94580>>
    feature_pixels = 16
    augment = True
    cache = False
    """

    def __init__(self, split, image_preprocessing_params, compute_feature_map_shape_fn, feature_pixels = 16, dir = "VOCdevkit/VOC2007", augment = True, shuffle = True, allow_difficult = False, cache = True):
        """
        Parameters
        ----------
        split : str
        Dataset split to load: train, val, or trainval.
        image_preprocessing_params : dataset.image.PreprocessingParams
        Image preprocessing parameters to apply when loading images.
        compute_feature_map_shape_fn : Callable[Tuple[int, int, int], Tuple[int, int, int]]
        Function to compute feature map shape, (channels, height, width), from
        input image shape, (channels, height, width).
        feature_pixels : int
        Size of each cell in the Faster R-CNN feature map in image pixels. This
        is the separation distance between anchors.
        dir : str
        Root directory of dataset.
        augment : bool
        Whether to randomly augment (horizontally flip) images during iteration
        with 50% probability.
        shuffle : bool
        Whether to shuffle the dataset each time it is iterated.
        allow_difficult : bool
        Whether to include ground truth boxes that are marked as "difficult".
        cache : bool
        Whether to training samples in memory after first being generated.
        """
        # import pdb; pdb.set_trace()
        if not os.path.exists(dir):
            raise FileNotFoundError("Dataset directory does not exist: %s" % dir)
        self.split = split
        self._dir = dir
        # self.class_index_to_name = self._get_classes()
        self.class_name_to_index = { class_name: class_index for (class_index, class_name) in self.class_index_to_name.items() }
        self.num_classes = len(self.class_index_to_name)
        assert self.num_classes == Dataset.num_classes, "Dataset does not have the expected number of classes (found %d but expected %d)" % (self.num_classes, Dataset.num_classes)
        assert self.class_index_to_name == Dataset.class_index_to_name, "Dataset does not have the expected class mapping"
        self.dataset_info = self._get_filepaths()
        self._filepaths = []
        self._gt_boxes_by_filepath = {}
        for idx in range(len(self.dataset_info['annotations'])):
            if idx in Dataset.invalid_anna_id[self.split]:
                continue
            img_id = self.dataset_info['annotations'][idx]['image_id']
            cur_file_name = self.dataset_info['images'][img_id]['file_name']
            category_id = self.dataset_info['annotations'][idx]['category_id']
            bbox = self.dataset_info['annotations'][idx]['bbox']
            if category_id == 0:    # 无瑕疵图
                corners = np.array([0, 0, 0, 0]).astype(np.float32)
                # corners = np.array([0, 0, 400, 400]).astype(np.float32)
                # continue
            else:
                corners = np.array([
                    bbox[0],
                    bbox[1],
                    bbox[0] + bbox[2],
                    bbox[1] + bbox[3]
                ]).astype(np.float32)
            if cur_file_name not in self._gt_boxes_by_filepath:
                cur_file_path = os.path.join(self._dir, self.split, 'sample', cur_file_name)
                self._gt_boxes_by_filepath[cur_file_path] = []
                self._filepaths.append(cur_file_path)
            box = Box(class_index=category_id,
                      class_name=self.class_index_to_name[category_id],
                      corners=corners)
            self._gt_boxes_by_filepath[cur_file_path].append(box)

        self.num_samples = len(self._filepaths)
        self._i = 0
        self._iterable_filepaths = self._filepaths.copy()
        self._image_preprocessing_params = image_preprocessing_params
        self._compute_feature_map_shape_fn = compute_feature_map_shape_fn
        self._feature_pixels = feature_pixels
        self._augment = augment
        self._shuffle = shuffle
        self._cache = cache
        self._unaugmented_cached_sample_by_filepath = {}
        self._augmented_cached_sample_by_filepath = {}

    def __iter__(self):
        self._i = 0
        if self._shuffle:
            random.shuffle(self._iterable_filepaths)
        return self

    def __next__(self):
        if self._i >= len(self._iterable_filepaths):
            raise StopIteration
        
        # Next file to load
        filepath = self._iterable_filepaths[self._i]
        self._i += 1

        # Augment?
        flip = random.randint(0, 1) != 0 if self._augment else 0
        cached_sample_by_filepath = self._augmented_cached_sample_by_filepath if flip else self._unaugmented_cached_sample_by_filepath

        # Load and, if caching, write back to cache
        if filepath in cached_sample_by_filepath:
            sample = cached_sample_by_filepath[filepath]
        else:
            sample = self._generate_training_sample(filepath = filepath, flip = flip)
        if self._cache:
            cached_sample_by_filepath[filepath] = sample

        # Return the sample
        return sample
    
    def _generate_training_sample(self, filepath, flip):
        # Load and preprocess the image
        # print(f"img path is {filepath}")
        scaled_image_data, scaled_image, scale_factor, original_shape = image.load_image(url = filepath, preprocessing = self._image_preprocessing_params, min_dimension_pixels = 600, horizontal_flip = flip)
        _, original_height, original_width = original_shape

        # Scale ground truth boxes to new image size
        scaled_gt_boxes = []
        for box in self._gt_boxes_by_filepath[filepath]:
            if flip:
                corners = np.array([
                box.corners[0],
                original_width - 1 - box.corners[3],
                box.corners[2],
                original_width - 1 - box.corners[1]
                ])
            else:
                corners = box.corners
            scaled_box = Box(
                class_index = box.class_index,
                class_name = box.class_name,
                corners = corners * scale_factor
            )
            scaled_gt_boxes.append(scaled_box)

        # Generate anchor maps and RPN truth map
        anchor_map, anchor_valid_map = anchors.generate_anchor_maps(image_shape = scaled_image_data.shape, feature_map_shape = self._compute_feature_map_shape_fn(scaled_image_data.shape), feature_pixels = self._feature_pixels)
        gt_rpn_map, gt_rpn_object_indices, gt_rpn_background_indices = anchors.generate_rpn_map(anchor_map = anchor_map, anchor_valid_map = anchor_valid_map, gt_boxes = scaled_gt_boxes)

        # Return sample
        return TrainingSample(
        anchor_map = anchor_map,
        anchor_valid_map = anchor_valid_map,
        gt_rpn_map = gt_rpn_map,
        gt_rpn_object_indices = gt_rpn_object_indices,
        gt_rpn_background_indices = gt_rpn_background_indices,
        gt_boxes = scaled_gt_boxes,
        image_data = scaled_image_data,
        image = scaled_image,
        filepath = filepath
        )

    def _get_filepaths(self):
        image_list_file = os.path.join(self._dir, self.split, self.split + ".json")
        with open(image_list_file) as fp:
            dataset_info = json.load(fp)
        return dataset_info
