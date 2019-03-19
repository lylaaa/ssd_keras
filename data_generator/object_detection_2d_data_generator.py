"""
A data generator for 2D object detection.

Copyright (C) 2018 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import division
import numpy as np
import inspect
from collections import defaultdict
import warnings
import sklearn.utils
from copy import deepcopy
from PIL import Image
import cv2
import csv
import os
import sys
from tqdm import tqdm, trange
import h5py
import json
from bs4 import BeautifulSoup
import pickle

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from data_generator.object_detection_2d_image_boxes_validation_utils import BoxFilter


class DegenerateBatchError(Exception):
    """
    An exception class to be raised if a generated batch ends up being degenerate,
    e.g. if a generated batch is empty.
    """
    pass


class DatasetError(Exception):
    """
    An exception class to be raised if a anything is wrong with the dataset,
    in particular if you try to generate batches when no dataset was loaded.
    """
    pass


class DataGenerator:
    """
    A generator to generate batches of samples and corresponding labels indefinitely.

    Can shuffle the dataset consistently after each complete pass.

    Currently provides three methods to parse annotation data:
        A general-purpose CSV parser,
        an XML parser for the Pascal VOC datasets,
        and a JSON parser for the MS COCO datasets.
    If the annotations of your dataset are in a format that is not supported by these parsers,
    you could just add another parser method and still use this generator.

    Can perform image transformations for data conversion and data augmentation,
    for details please refer to the documentation of the `generate()` method.
    """

    def __init__(self,
                 load_images_into_memory=False,
                 hdf5_dataset_path=None,
                 filenames=None,
                 filenames_type='text',
                 images_dir=None,
                 labels=None,
                 image_ids=None,
                 eval_neutral=None,
                 labels_output_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax'),
                 verbose=True):
        """
        Initializes the data generator. You can either load a dataset directly here in the constructor,
        e.g. an HDF5 dataset, or you can use one of the parser methods to read in a dataset.

        Arguments:
            load_images_into_memory (bool, optional): If `True`, the entire dataset will be loaded into memory.
                This enables noticeably faster data generation than loading batches of images into memory ad hoc.
                Be sure that you have enough memory before you activate this option.
            hdf5_dataset_path (str, optional): The full file path of an HDF5 file that contains a dataset in the
                format that the `create_hdf5_dataset()` method produces. If you load such an HDF5 dataset, you
                don't need to use any of the parser methods anymore, the HDF5 dataset already contains all relevant
                data.
            filenames (string or list, optional): `None` or either a Python list/tuple or a string representing
                a filepath. If a list/tuple is passed, it must contain the file names (full paths) of the
                images to be used. Note that the list/tuple must contain the paths to the images,
                not the images themselves. If a filepath string is passed, it must point either to
                (1) a pickled file containing a list/tuple as described above. In this case the `filenames_type`
                argument must be set to `pickle`.
                Or
                (2) a text file. Each line of the text file contains the file name (basename of the file only,
                not the full directory path) to one image and nothing else. In this case the `filenames_type`
                argument must be set to `text` and you must pass the path to the directory that contains the
                images in `images_dir`.
            filenames_type (string, optional): In case a string is passed for `filenames`, this indicates what
                type of file `filenames` is. It can be either 'pickle' for a pickled file or 'text' for a
                plain text file.
            images_dir (string, optional): In case a text file is passed for `filenames`, the full paths to
                the images will be composed from `images_dir` and the names in the text file, i.e. this
                should be the directory that contains the images to which the text file refers.
                If `filenames_type` is not 'text', then this argument is irrelevant.
            labels (string or list, optional): `None` or either a Python list/tuple or a string representing
                the path to a pickled file containing a list/tuple. The list/tuple must contain Numpy arrays
                that represent the labels of the dataset.
            image_ids (string or list, optional): `None` or either a Python list/tuple or a string representing
                the path to a pickled file containing a list/tuple. The list/tuple must contain the image
                IDs of the images in the dataset.
            eval_neutral (string or list, optional): `None` or either a Python list/tuple or a string representing
                the path to a pickled file containing a list/tuple. The list/tuple must contain for each image
                a list that indicates for each ground truth object in the image whether that object is supposed
                to be treated as neutral during an evaluation.
            labels_output_format (list, optional): A list of five strings representing the desired order of the five
                items class ID, xmin, ymin, xmax, ymax in the generated ground truth data (if any). The expected
                strings are 'xmin', 'ymin', 'xmax', 'ymax', 'class_id'.
            verbose (bool, optional): If `True`, prints out the progress for some constructor operations that may
                take a bit longer.
        """
        self.labels_output_format = labels_output_format
        # As long as we haven't loaded anything yet, the dataset size is zero.
        self.dataset_size = 0
        self.load_images_into_memory = load_images_into_memory
        # The only way that this list will not stay `None` is if `load_images_into_memory == True`.
        self.images = None

        # `self.filenames` is a list containing all file names of the image samples (full paths).
        # Note that it does not contain the actual image files themselves.
        #  This list is one of the outputs of the parser methods.
        # In case you are loading an HDF5 dataset, this list will be `None`.
        if filenames is not None:
            if isinstance(filenames, (list, tuple)):
                self.filenames = filenames
            elif isinstance(filenames, str):
                    if filenames_type == 'pickle':
                        # Adam
                        with open(filenames, 'rb') as f:
                            self.filenames = pickle.load(f)
                    elif filenames_type == 'text':
                        if images_dir is not None:
                            with open(filenames, 'r') as f:
                                self.filenames = [os.path.join(images_dir, line.strip()) for line in f]
                        else:
                            raise ValueError("`images_dir` can not be None when filenames_type has been set as 'text'")
                    else:
                        raise ValueError("`filenames_type` can be either 'text' or 'pickle'.")
            else:
                raise ValueError(
                    "`filenames` must be either a Python list/tuple"
                    "or a string representing a filepath (to a pickled or text file)."
                    "The value you passed is neither of the two.")
            self.dataset_size = len(self.filenames)
            self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
            if load_images_into_memory:
                self.images = []
                if verbose:
                    it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
                else:
                    it = self.filenames
                for filename in it:
                    with Image.open(filename) as image:
                        self.images.append(np.array(image, dtype=np.uint8))
        else:
            self.filenames = None

        # In case ground truth is available, `self.labels` is a list containing for each image a list (or NumPy array)
        # of ground truth bounding boxes for that image.
        if labels is not None:
            if isinstance(labels, str):
                with open(labels, 'rb') as f:
                    self.labels = pickle.load(f)
            elif isinstance(labels, (list, tuple)):
                self.labels = labels
            else:
                raise ValueError(
                    "`labels` must be either a Python list/tuple"
                    "or a string representing the path to a pickled file containing a list/tuple."
                    "The value you passed is neither of the two.")
        else:
            self.labels = None

        if image_ids is not None:
            if isinstance(image_ids, str):
                with open(image_ids, 'rb') as f:
                    self.image_ids = pickle.load(f)
            elif isinstance(image_ids, (list, tuple)):
                self.image_ids = image_ids
            else:
                raise ValueError(
                    "`image_ids` must be either a Python list/tuple"
                    "or a string representing the path to a pickled file containing a list/tuple."
                    "The value you passed is neither of the two.")
        else:
            self.image_ids = None

        if eval_neutral is not None:
            if isinstance(eval_neutral, str):
                with open(eval_neutral, 'rb') as f:
                    self.eval_neutral = pickle.load(f)
            elif isinstance(eval_neutral, (list, tuple)):
                self.eval_neutral = eval_neutral
            else:
                raise ValueError(
                    "`eval_neutral` must be either a Python list/tuple"
                    "or a string representing the path to a pickled file containing a list/tuple."
                    "The value you passed is neither of the two.")
        else:
            self.eval_neutral = None

        if hdf5_dataset_path is not None:
            self.hdf5_dataset_path = hdf5_dataset_path
            self.load_hdf5_dataset(verbose=verbose)
        else:
            self.hdf5_dataset = None

    def load_hdf5_dataset(self, verbose=True):
        """
        Loads an HDF5 dataset that is in the format that the `create_hdf5_dataset()` method produces.

        Arguments:
            verbose (bool, optional): If `True`, prints out the progress while loading the dataset.

        Returns:
            None.
        """

        self.hdf5_dataset = h5py.File(self.hdf5_dataset_path, 'r')
        self.dataset_size = len(self.hdf5_dataset['images'])
        # Instead of shuffling the HDF5 dataset or images in memory, we will shuffle this index list.
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)

        if self.load_images_into_memory:
            self.images = []
            if verbose:
                tr = trange(self.dataset_size, desc='Loading images into memory', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.images.append(self.hdf5_dataset['images'][i].reshape(self.hdf5_dataset['image_shapes'][i]))

        if self.hdf5_dataset.attrs['has_labels']:
            self.labels = []
            labels = self.hdf5_dataset['labels']
            label_shapes = self.hdf5_dataset['label_shapes']
            if verbose:
                tr = trange(self.dataset_size, desc='Loading labels', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.labels.append(labels[i].reshape(label_shapes[i]))

        if self.hdf5_dataset.attrs['has_image_ids']:
            self.image_ids = []
            image_ids = self.hdf5_dataset['image_ids']
            if verbose:
                tr = trange(self.dataset_size, desc='Loading image IDs', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.image_ids.append(image_ids[i])

        if self.hdf5_dataset.attrs['has_eval_neutral']:
            self.eval_neutral = []
            eval_neutral = self.hdf5_dataset['eval_neutral']
            if verbose:
                tr = trange(self.dataset_size, desc='Loading evaluation-neutrality annotations', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.eval_neutral.append(eval_neutral[i])

    def parse_csv(self,
                  images_dir,
                  labels_filename,
                  input_format,
                  include_classes='all',
                  random_sample=False,
                  ret=False,
                  verbose=True):
        """
        Arguments:
            images_dir (str): The path to the directory that contains the images.
            labels_filename (str): The filepath to a CSV file that contains one ground truth bounding box per line
                and each line contains the following six items: image file name, class ID, xmin, xmax, ymin, ymax.
                The six items do not have to be in a specific order, but they must be the first six columns of
                each line. The order of these items in the CSV file must be specified in `input_format`.
                The class ID is an integer greater than zero. Class ID 0 is reserved for the background class.
                `xmin` and `xmax` are the left-most and right-most absolute horizontal coordinates of the box,
                `ymin` and `ymax` are the top-most and bottom-most absolute vertical coordinates of the box.
                The image name is expected to be just the name of the image file without the directory path
                at which the image is located.
            input_format (list): A list of six strings representing the order of the six items
                image file name, class ID, xmin, xmax, ymin, ymax in the input CSV file.
                For udacity self-driving dataset, the expected list is ['image_name', 'xmin', 'xmax', 'ymin', 'ymax',
                'class_id'].
            include_classes (list, optional): Either 'all' or a list of integers containing the class IDs that
                are to be included in the dataset. If 'all', all ground truth boxes will be included in the dataset.
            random_sample (float, optional): Either `False` or a float in `[0,1]`.
                If this is `False`, the full dataset will be used by the generator.
                If this is a float in `[0,1]`, a randomly sampled fraction of the dataset will be used, where
                `random_sample` is the fraction of the dataset to be used.
                For example, if `random_sample = 0.2`, 20 percent of the dataset will be randomly selected,
                the rest will be omitted.
                The fraction refers to the number of images, not to the number of boxes,
                i.e. each image that will be added to the dataset will always be added with all of its boxes.
            ret (bool, optional): Whether or not to return the outputs of the parser.
            verbose (bool, optional): If `True`, prints out the progress for operations that may take a bit longer.

        Returns:
            None by default,
            optionally lists for whichever are available of images, image filenames, labels, and image IDs.
        """

        # Set class members.
        self.images_dir = images_dir
        self.labels_filename = labels_filename
        self.input_format = input_format
        self.include_classes = include_classes

        # Before we begin, make sure that we have a labels_filename and an input_format
        if self.labels_filename is None or self.input_format is None:
            raise ValueError(
                "`labels_filename` and/or `input_format` have not been set yet. You need to pass them as arguments.")

        # Erase data that might have been parsed before
        self.filenames = []
        self.image_ids = []
        self.labels = []

        # First, just read in the CSV file lines and sort them.
        data = []
        # newline='' 表示认为 '\n', '\r', or '\r\n' 为换行符, 但是不对它们进行转换, 默认是转换成 '\n'
        with open(self.labels_filename, newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            # Skip the header row.
            next(csv_reader)
            # For every line (i.e for every bounding box) in the CSV file...
            for row in csv_reader:
                # If the class_id is among the classes that are to be included in the dataset...
                if self.include_classes == 'all' or \
                        int(row[self.input_format.index('class_id')].strip()) in self.include_classes:
                    # Store the box class and coordinates here
                    box = []
                    # Select the image name column in the input format and append its content to `box`
                    box.append(row[self.input_format.index('image_name')].strip())
                    # For each element in the output format
                    # defaults: the elements are the class ID and the four box coordinates (xmin,ymin,xmax,ymax)
                    for element in self.labels_output_format:
                        # select the respective column in the input format and append it to `box`.
                        box.append(int(row[self.input_format.index(element)].strip()))
                    data.append(box)

        # The data needs to be sorted, otherwise the next step won't give the correct result
        data = sorted(data)

        # Now that we've made sure that the data is sorted by file names,
        # we can compile the actual samples and labels lists

        # The current image for which we're collecting the ground truth boxes
        current_file = data[0][0]
        # The image ID will be the portion of the image name before the first dot.
        current_image_id = data[0][0].split('.')[0]
        # The list where we collect all ground truth boxes for a given image
        current_labels = []
        for i, box in enumerate(data):
            # If this box (i.e. this line of the CSV file) belongs to the current image file
            if box[0] == current_file:
                current_labels.append(box[1:])
                # If this is the last box
                if i == len(data) - 1:
                    # In case we're not using the full dataset, but a random sample of it.
                    if random_sample:
                        p = np.random.uniform(0, 1)
                        if p >= (1 - random_sample):
                            self.labels.append(np.stack(current_labels, axis=0))
                            self.filenames.append(os.path.join(self.images_dir, current_file))
                            self.image_ids.append(current_image_id)
                    else:
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)
            # If this box belongs to a new image file
            else:
                # In case we're not using the full dataset, but a random sample of it.
                # 是否把上一个 image 加入 dataset
                if random_sample:
                    p = np.random.uniform(0, 1)
                    if p >= (1 - random_sample):
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)
                else:
                    self.labels.append(np.stack(current_labels, axis=0))
                    self.filenames.append(os.path.join(self.images_dir, current_file))
                    self.image_ids.append(current_image_id)
                # Reset the labels list because this is a new file.
                current_labels = []
                current_file = box[0]
                current_image_id = box[0].split('.')[0]
                current_labels.append(box[1:])
                # If this is the last line box
                if i == len(data) - 1:
                    # In case we're not using the full dataset, but a random sample of it.
                    if random_sample:
                        p = np.random.uniform(0, 1)
                        if p >= (1 - random_sample):
                            self.labels.append(np.stack(current_labels, axis=0))
                            self.filenames.append(os.path.join(self.images_dir, current_file))
                            self.image_ids.append(current_image_id)
                    else:
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)

        self.dataset_size = len(self.filenames)
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
        if self.load_images_into_memory:
            self.images = []
            if verbose:
                it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
            else:
                it = self.filenames
            for filename in it:
                with Image.open(filename) as image:
                    self.images.append(np.array(image, dtype=np.uint8))
        # In case we want to return these
        if ret:
            return self.images, self.filenames, self.labels, self.image_ids

    def parse_xml(self,
                  images_dirs,
                  image_set_filenames,
                  annotations_dirs=(),
                  classes=('background',
                           'aeroplane', 'bicycle', 'bird', 'boat',
                           'bottle', 'bus', 'car', 'cat',
                           'chair', 'cow', 'diningtable', 'dog',
                           'horse', 'motorbike', 'person', 'pottedplant',
                           'sheep', 'sofa', 'train', 'tvmonitor'),
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False,
                  verbose=True):
        """
        This is an XML parser for the Pascal VOC datasets. It might be applicable to other datasets with minor changes
        to the code, but in its current form it expects the data format and XML tags of the Pascal VOC datasets.

        Arguments:
            images_dirs (list): A list of strings, where each string is the path of a directory that
                contains images that are to be part of the dataset. This allows you to aggregate multiple datasets
                into one (e.g. one directory that contains the images for Pascal VOC 2007, another that contains
                the images for Pascal VOC 2012, etc.).
            image_set_filenames (list): A list of strings, where each string is the path of the text file with the image
                set to be loaded. Must be one file per image directory given. These text files define what images in the
                respective image directories are to be part of the dataset and simply contains one image ID per line
                and nothing else.
            annotations_dirs (list, optional): A list of strings, where each string is the path of a directory that
                contains the annotations (XML files) that belong to the images in the respective image directories given.
                The directories must contain one XML file per image and the name of an XML file must be the image ID
                of the image it belongs to. The content of the XML files must be in the Pascal VOC format.
            classes (list, optional): A list containing the names of the object classes as found in the
                `name` XML tags. Must include the class `background` as the first list item.
                The order of this list defines the class IDs.
            include_classes (list, optional): Either 'all' or a list of integers containing the class IDs that
                are to be included in the dataset.
                If 'all', all ground truth boxes will be included in the dataset.
            exclude_truncated (bool, optional): If `True`, excludes boxes that are labeled as 'truncated'.
            exclude_difficult (bool, optional): If `True`, excludes boxes that are labeled as 'difficult'.
            ret (bool, optional): Whether or not to return the outputs of the parser.
            verbose (bool, optional): If `True`, prints out the progress for operations that may take a bit longer.

        Returns:
            None by default, optionally lists for whichever are available of images, image filenames, labels, image IDs,
            and a list indicating which boxes are annotated with the label "difficult".
        """
        # Set class members.
        self.images_dirs = images_dirs
        self.annotations_dirs = annotations_dirs
        self.image_set_filenames = image_set_filenames
        self.classes = classes
        self.include_classes = include_classes

        # Erase data that might have been parsed before.
        self.filenames = []
        self.image_ids = []
        self.labels = []
        self.eval_neutral = []
        if not annotations_dirs:
            self.labels = None
            self.eval_neutral = None
            annotations_dirs = [None] * len(images_dirs)

        for images_dir, image_set_filename, annotations_dir in zip(images_dirs, image_set_filenames, annotations_dirs):
            # Read the image set file that so that we know all the IDs of all the images to be included in the dataset.
            with open(image_set_filename) as f:
                # Note: These are strings, not integers.
                image_ids = [line.strip() for line in f]
                self.image_ids += image_ids

            if verbose:
                it = tqdm(image_ids, desc="Processing image set '{}'".format(os.path.basename(image_set_filename)),
                          file=sys.stdout)
            else:
                it = image_ids

            # Loop over all images in this dataset.
            for image_id in it:
                filename = '{}'.format(image_id) + '.jpg'
                self.filenames.append(os.path.join(images_dir, filename))
                if annotations_dir is not None:
                    # Parse the XML file for this image.
                    with open(os.path.join(annotations_dir, image_id + '.xml')) as f:
                        soup = BeautifulSoup(f, 'xml')
                    # In case we want to return the folder in addition to the image file name.
                    # Relevant for determining which dataset an image belongs to.
                    folder = soup.folder.text
                    # filename = soup.filename.text
                    # We'll store all boxes for this image here.
                    boxes = []
                    # We'll store whether a box is annotated as "difficult" here.
                    eval_neutr = []
                    # Get a list of all objects in this image.
                    objects = soup.find_all('object')

                    # Parse the data for each object.
                    for obj in objects:
                        class_name = obj.find('name', recursive=False).text
                        class_id = self.classes.index(class_name)
                        # Check whether this class is supposed to be included in the dataset.
                        if (self.include_classes != 'all') and (class_id not in self.include_classes):
                            continue
                        pose = obj.find('pose', recursive=False).text
                        truncated = int(obj.find('truncated', recursive=False).text)
                        if exclude_truncated and (truncated == 1):
                            continue
                        difficult = int(obj.find('difficult', recursive=False).text)
                        if exclude_difficult and (difficult == 1):
                            continue
                        # Get the bounding box coordinates.
                        bndbox = obj.find('bndbox', recursive=False)
                        xmin = int(bndbox.xmin.text)
                        ymin = int(bndbox.ymin.text)
                        xmax = int(bndbox.xmax.text)
                        ymax = int(bndbox.ymax.text)
                        item_dict = {'folder': folder,
                                     'image_name': filename,
                                     'image_id': image_id,
                                     'class_name': class_name,
                                     'class_id': class_id,
                                     'pose': pose,
                                     'truncated': truncated,
                                     'difficult': difficult,
                                     'xmin': xmin,
                                     'ymin': ymin,
                                     'xmax': xmax,
                                     'ymax': ymax}
                        box = []
                        for item in self.labels_output_format:
                            box.append(item_dict[item])
                        boxes.append(box)
                        if difficult:
                            eval_neutr.append(True)
                        else:
                            eval_neutr.append(False)
                    # TODO: 确认要不要转成 np.array
                    self.labels.append(boxes)
                    self.eval_neutral.append(eval_neutr)

        self.dataset_size = len(self.filenames)
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
        if self.load_images_into_memory:
            self.images = []
            if verbose:
                it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
            else:
                it = self.filenames
            for filename in it:
                with Image.open(filename) as image:
                    self.images.append(np.array(image, dtype=np.uint8))

        if ret:
            return self.images, self.filenames, self.labels, self.image_ids, self.eval_neutral

    def parse_json(self,
                   images_dirs,
                   annotations_filenames,
                   ground_truth_available=False,
                   include_classes='all',
                   ret=False,
                   verbose=True):
        """
        This is an JSON parser for the MS COCO datasets. It might be applicable to other datasets with minor changes to
        the code, but in its current form it expects the JSON format of the MS COCO datasets.

        Arguments:
            images_dirs (list, optional): A list of strings, where each string is the path of a directory that
                contains images that are to be part of the dataset. This allows you to aggregate multiple datasets
                into one (e.g. one directory that contains the images for MS COCO Train 2014, another one for MS COCO
                Val 2014, another one for MS COCO Train 2017 etc.).
            annotations_filenames (list): A list of strings, where each string is the path of the JSON file
                that contains the annotations for the images in the respective image directories given, i.e. one
                JSON file per image directory that contains the annotations for all images in that directory.
                The content of the JSON files must be in MS COCO object detection format.
                Note that these annotations files do not necessarily need to contain ground truth information.
                MS COCO also provides annotations files without ground truth information for the test datasets,
                called `image_info_[...].json`.
            ground_truth_available (bool, optional): Set `True` if the annotations files contain ground truth information.
            include_classes (list, optional): Either 'all' or a list of integers containing the class IDs that
                are to be included in the dataset.
                If 'all', all ground truth boxes will be included in the dataset.
            ret (bool, optional): Whether or not to return the outputs of the parser.
            verbose (bool, optional): If `True`, prints out the progress for operations that may take a bit longer.

        Returns:
            None by default, optionally lists for whichever are available of images, image filenames, labels and image IDs.
        """
        self.images_dirs = images_dirs
        self.annotations_filenames = annotations_filenames
        self.include_classes = include_classes
        # Erase data that might have been parsed before.
        self.filenames = []
        self.image_ids = []
        self.labels = []
        if not ground_truth_available:
            self.labels = None

        # Build the dictionaries that map between class names and class IDs.
        with open(annotations_filenames[0], 'r') as f:
            annotations = json.load(f)
        # annotations 的内容格式参见 http://cocodataset.org/#format-data
        # 以 /home/adam/.keras/datasets/coco/2014_80_40/annotations/instances/instances_val2014.json 为例:
        # annotations 的 key 有 ['info', 'images', 'licenses', 'annotations', 'categories']
        # Unfortunately the 80 MS COCO class IDs are not all consecutive.
        # They go from 1 to 90 and some numbers are skipped.
        # Since the IDs that we feed into a neural network must be consecutive,
        # we'll save both the original (non-consecutive) IDs as well as transformed maps.
        # We'll save both the map between the original
        # The map between class names (values) and their original IDs (keys)
        self.cats_to_names = {}
        # A list of the class names with their indices representing the transformed IDs
        self.classes_to_names = []
        # Need to add the background class first so that the indexing is right.
        self.classes_to_names.append('background')
        # A dictionary that maps between the original (keys) and the transformed IDs (values)
        self.cats_to_classes = {}
        # A dictionary that maps between the transformed (keys) and the original IDs (values)
        self.classes_to_cats = {}
        for i, cat in enumerate(annotations['categories']):
            self.cats_to_names[cat['id']] = cat['name']
            self.classes_to_names.append(cat['name'])
            self.cats_to_classes[cat['id']] = i + 1
            self.classes_to_cats[i + 1] = cat['id']

        # Iterate over all datasets.
        for images_dir, annotations_filename in zip(self.images_dirs, self.annotations_filenames):
            # Load the JSON file.
            with open(annotations_filename, 'r') as f:
                annotations = json.load(f)

            if ground_truth_available:
                # Create the annotations map, a dictionary whose keys are the image IDs
                # and whose values are the annotations for the respective image ID.
                image_ids_to_annotations = defaultdict(list)
                for annotation in annotations['annotations']:
                    # Note: 一个 image_id 会对应多个 annotation
                    image_ids_to_annotations[annotation['image_id']].append(annotation)

            # annotations['images'] 是一个元素为 json 类型的 list
            # 每一个元素的样式为:
            # {'license': 3,
            #  'file_name': 'COCO_val2014_000000391895.jpg',
            #  'coco_url': 'http://mscoco.org/images/391895',
            #  'height': 360,
            #  'width': 640,
            #  'date_captured': '2013-11-14 11:18:45',
            #  'flickr_url': 'http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg',
            #  'id': 391895}
            if verbose:
                it = tqdm(annotations['images'], desc="Processing '{}'".
                          format(os.path.basename(annotations_filename)), file=sys.stdout)
            else:
                it = annotations['images']

            # Loop over all images in this dataset.
            for image in it:
                self.filenames.append(os.path.join(images_dir, image['file_name']))
                self.image_ids.append(image['id'])
                if ground_truth_available:
                    # Get all annotations for this image.
                    annotations = image_ids_to_annotations[image['id']]
                    boxes = []
                    for annotation in annotations:
                        cat_id = annotation['category_id']
                        # Check if this class is supposed to be included in the dataset.
                        if (self.include_classes != 'all') and (cat_id not in self.include_classes):
                            continue
                        # Transform the original class ID to fit in the sequence of consecutive IDs.
                        class_id = self.cats_to_classes[cat_id]
                        xmin = annotation['bbox'][0]
                        ymin = annotation['bbox'][1]
                        width = annotation['bbox'][2]
                        height = annotation['bbox'][3]
                        # Compute `xmax` and `ymax`.
                        xmax = xmin + width
                        ymax = ymin + height
                        item_dict = {'image_name': image['file_name'],
                                     'image_id': image['id'],
                                     'class_id': class_id,
                                     'xmin': xmin,
                                     'ymin': ymin,
                                     'xmax': xmax,
                                     'ymax': ymax}
                        box = []
                        for item in self.labels_output_format:
                            box.append(item_dict[item])
                        boxes.append(box)
                    self.labels.append(boxes)

        self.dataset_size = len(self.filenames)
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
        if self.load_images_into_memory:
            self.images = []
            if verbose:
                it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
            else:
                it = self.filenames
            for filename in it:
                with Image.open(filename) as image:
                    self.images.append(np.array(image, dtype=np.uint8))

        if ret:
            return self.images, self.filenames, self.labels, self.image_ids

    def create_hdf5_dataset(self,
                            file_path='dataset.h5',
                            resize=False,
                            variable_image_size=True,
                            verbose=True):
        """
        Converts the currently loaded dataset into a HDF5 file.
        This HDF5 file contains all images as uncompressed arrays in a contiguous block of memory, which
        allows for them to be loaded faster.
        Such an uncompressed dataset, however, may take up considerably more space on your hard drive than
        the sum of the source images in a compressed format such as JPG or PNG.

        It is recommended that you always convert the dataset into an HDF5 dataset if you
        have enough hard drive space since loading from an HDF5 dataset accelerates the data generation noticeably.

        Note that you must load a dataset (e.g. via one of the parser methods) before creating an HDF5 dataset from it.

        The created HDF5 dataset will remain open upon its creation so that it can be used right away.

        Arguments:
            file_path (str, optional): The full file path under which to store the HDF5 dataset.
                You can load this output file via the `DataGenerator` constructor in the future.
            resize (tuple, optional): `False` or a 2-tuple `(height, width)` that represents the
                target size for the images. All images in the dataset will be resized to this
                target size before they will be written to the HDF5 file. If `False`, no resizing
                will be performed.
            variable_image_size (bool, optional): The only purpose of this argument is that its
                value will be stored in the HDF5 dataset in order to be able to quickly find out
                whether the images in the dataset all have the same size or not.
            verbose (bool, optional): Whether or not print out the progress of the dataset creation.

        Returns:
            None.
        """
        dataset_size = len(self.filenames)

        # Create the HDF5 file.
        hdf5_dataset = h5py.File(file_path, 'w')

        # Create a few attributes that tell us what this dataset contains.
        # The dataset will obviously always contain images, but maybe it will
        # also contain labels, image IDs, etc.
        hdf5_dataset.attrs.create(name='has_labels', data=False, shape=None, dtype=np.bool_)
        hdf5_dataset.attrs.create(name='has_image_ids', data=False, shape=None, dtype=np.bool_)
        hdf5_dataset.attrs.create(name='has_eval_neutral', data=False, shape=None, dtype=np.bool_)
        # It's useful to be able to quickly check whether the images in a dataset all
        # have the same size or not, so add a boolean attribute for that.
        if variable_image_size and not resize:
            hdf5_dataset.attrs.create(name='variable_image_size', data=True, shape=None, dtype=np.bool_)
        else:
            hdf5_dataset.attrs.create(name='variable_image_size', data=False, shape=None, dtype=np.bool_)

        # Create the dataset in which the images will be stored as flattened arrays.
        # This allows us, among other things, to store images of variable size.
        hdf5_images = hdf5_dataset.create_dataset(name='images',
                                                  shape=(dataset_size,),
                                                  maxshape=None,
                                                  dtype=h5py.special_dtype(vlen=np.uint8))

        # Create the dataset that will hold the image heights, widths and channels that
        # we need in order to reconstruct the images from the flattened arrays later.
        hdf5_image_shapes = hdf5_dataset.create_dataset(name='image_shapes',
                                                        shape=(dataset_size, 3),
                                                        maxshape=(None, 3),
                                                        dtype=np.int32)

        if self.labels is not None:
            # Create the dataset in which the labels will be stored as flattened arrays.
            hdf5_labels = hdf5_dataset.create_dataset(name='labels',
                                                      shape=(dataset_size,),
                                                      maxshape=None,
                                                      dtype=h5py.special_dtype(vlen=np.int32))

            # Create the dataset that will hold the dimensions of the labels arrays for
            # each image so that we can restore the labels from the flattened arrays later.
            hdf5_label_shapes = hdf5_dataset.create_dataset(name='label_shapes',
                                                            shape=(dataset_size, 2),
                                                            maxshape=(None, 2),
                                                            dtype=np.int32)

            hdf5_dataset.attrs.modify(name='has_labels', value=True)

        if self.image_ids is not None:
            hdf5_image_ids = hdf5_dataset.create_dataset(name='image_ids',
                                                         shape=(dataset_size,),
                                                         maxshape=None,
                                                         dtype=h5py.special_dtype(vlen=str))

            hdf5_dataset.attrs.modify(name='has_image_ids', value=True)

        if self.eval_neutral is not None:
            # Create the dataset in which the labels will be stored as flattened arrays.
            hdf5_eval_neutral = hdf5_dataset.create_dataset(name='eval_neutral',
                                                            shape=(dataset_size,),
                                                            maxshape=None,
                                                            dtype=h5py.special_dtype(vlen=np.bool_))

            hdf5_dataset.attrs.modify(name='has_eval_neutral', value=True)

        if verbose:
            tr = trange(dataset_size, desc='Creating HDF5 dataset', file=sys.stdout)
        else:
            tr = range(dataset_size)

        # Iterate over all images in the dataset.
        for i in tr:
            # Store the image.
            with Image.open(self.filenames[i]) as image:
                image = np.asarray(image, dtype=np.uint8)
                # Make sure all images end up having three channels.
                # 且最后一个 axis 的维度为 3
                if image.ndim == 2:
                    image = np.stack([image] * 3, axis=-1)
                elif image.ndim == 3:
                    if image.shape[2] == 1:
                        image = np.concatenate([image] * 3, axis=-1)
                    elif image.shape[2] == 4:
                        image = image[:, :, :3]

                if resize:
                    image = cv2.resize(image, dsize=(resize[1], resize[0]))

                # Flatten the image array and write it to the images dataset.
                hdf5_images[i] = image.reshape(-1)
                # Write the image's shape to the image shapes dataset.
                hdf5_image_shapes[i] = image.shape

            # Store the ground truth if we have any.
            if self.labels is not None:
                labels = np.asarray(self.labels[i])
                # Flatten the labels array and write it to the labels dataset.
                hdf5_labels[i] = labels.reshape(-1)
                # Write the labels' shape to the label shapes dataset.
                hdf5_label_shapes[i] = labels.shape

            # Store the image ID if we have one.
            if self.image_ids is not None:
                hdf5_image_ids[i] = self.image_ids[i]

            # Store the evaluation-neutrality annotations if we have any.
            if self.eval_neutral is not None:
                hdf5_eval_neutral[i] = self.eval_neutral[i]

        hdf5_dataset.close()
        self.hdf5_dataset = h5py.File(file_path, 'r')
        self.hdf5_dataset_path = file_path
        self.dataset_size = len(self.hdf5_dataset['images'])
        # Instead of shuffling the HDF5 dataset, we will shuffle this index list.
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)

    def generate(self,
                 batch_size=32,
                 shuffle=True,
                 transformations=(),
                 label_encoder=None,
                 returns=('processed_images', 'encoded_labels'),
                 keep_images_without_gt=False,
                 degenerate_box_handling='remove'):
        """
        Generates batches of samples and (optionally) corresponding labels indefinitely.
        Can shuffle the samples consistently after each complete pass.
        Optionally takes a list of arbitrary image transformations to apply to the samples ad hoc(临时).

        Arguments:
            batch_size (int, optional): The size of the batches to be generated.
            shuffle (bool, optional): Whether or not to shuffle the dataset before each pass.
                This option should always be `True` during training, but it can be useful to turn shuffling off
                for debugging or if you're using the generator for prediction.
            transformations (tuple, optional): A tuple of transformations that will be applied to the images and labels
                in the given order.
                Each transformation is a callable that takes as input an image (as a Numpy array) and
                optionally labels (also as a Numpy array) and returns an image and optionally labels in the same format.
            label_encoder (callable, optional): Only relevant if labels are given. A callable that takes as input the
                labels of a batch (as a list of Numpy arrays) and returns some structure that represents those labels.
                The general use case for this is to convert labels from their input format to a format that a given
                object detection model needs as its training targets.
            returns (tuple, optional): A tuple of strings that determines what outputs the generator yields.
                The generator's output is always a tuple that contains the outputs specified in this tuple and only
                those.
                If an output is not available, it will be `None`.
                The output tuple can contain the following outputs according to the specified keyword strings:
                * 'processed_images': An array containing the processed images. Will always be in the outputs, so
                    it doesn't matter whether or not you include this keyword in the set.
                * 'encoded_labels': The encoded labels tensor. Will always be in the outputs if a label encoder is
                    given, so it doesn't matter whether or not you include this keyword in the set if you pass a label
                    encoder.
                * 'matched_anchors': Only available if `labels_encoder` is an `SSDInputEncoder` object. The same as
                    'encoded_labels', but containing anchor box coordinates for all matched anchor boxes instead of
                    ground truth coordinates.
                    This can be useful to visualize what anchor boxes are being matched to each ground truth box.
                    Only available in training mode.
                * 'processed_labels': The processed, but not yet encoded labels. This is a list that contains for each
                    batch image a Numpy array with all ground truth boxes for that image.
                    Only available if ground truth is available.
                * 'filenames': A list containing the file names (full paths) of the images in the batch.
                * 'image_ids': A list containing the integer IDs of the images in the batch. Only available if there
                    are image IDs available.
                * 'evaluation_neutral': A nested list of lists of booleans. Each list contains `True` or `False` for
                    every ground truth bounding box of the respective image depending on whether that
                    bounding box is supposed to be evaluation-neutral (`True`) or not (`False`).
                    May return `None` if there exists no such concept for a given dataset. An example for
                    evaluation_neutrality is the ground truth boxes annotated as "difficult" in the Pascal VOC
                    datasets, which are usually treated to be neutral in a model evaluation.
                * 'inverse_transform': A nested list that contains a list of "inverter" functions for each item in the
                    batch.
                    These inverter functions take (predicted) labels for an image as input and apply the inverse of the
                    transformations that were applied to the original image to them. This makes it possible to
                    let the model make predictions on a transformed image and then convert these predictions
                    back to the original image.
                    This is mostly relevant for evaluation.
                    If you want to evaluate your model on a dataset with varying image sizes, then you are forced to
                    transform the images somehow (e.g. by resizing or cropping) to make them all the same size.
                    Your model will then predict boxes for those transformed images, but for the evaluation you will
                    need predictions with respect to the original images, not with respect to the transformed images.
                    This means you will have to transform the predicted box coordinates back to the original image size.
                    Note that for each image, the inverter functions for that image need to be applied in the <inverse?>
                     order in which they are given in the respective list for that image.
                * 'original_images': A list containing the original images in the batch before any processing.
                * 'original_labels': A list containing the original ground truth boxes for the images in this batch
                    before any processing. Only available if ground truth is available.
                The order of the outputs in the tuple is the order of the tuple above. If `returns` contains a keyword
                for an output that is unavailable,that output omitted in the yielded tuple and a warning will be raised.
            keep_images_without_gt (bool, optional): If `False`, images for which there aren't any ground truth boxes
                before any transformations have been applied will be removed from the batch. If `True`, such images will
                be kept in the batch.
            degenerate_box_handling (str, optional): How to handle degenerate boxes, which are boxes that have
                `xmax <= xmin` and/or `ymax <= ymin`. Degenerate boxes can sometimes be in the dataset, or
                non-degenerate boxes can become degenerate after they were processed by transformations.
                Note that the generator checks for degenerate boxes after all transformations have been applied if any,
                 but before the labels were passed to the `label_encoder` (if one was given).
                Can be one of 'warn' or 'remove'.
                If 'warn', the generator will merely print a warning to let you know that there are degenerate boxes in
                a batch.
                If 'remove', the generator will remove degenerate boxes from the batch silently.
        Yields:
            The next batch as a tuple of items as defined by the `returns` argument.
        """
        if self.dataset_size == 0:
            raise DatasetError("Cannot generate batches because you did not load a dataset.")

        if degenerate_box_handling not in ['remove', 'warn']:
            raise ValueError("`degenerate_box_handling` must be either 'remove' or 'warn'")

        #############################################################################################
        # Warn if any of the set returns aren't possible.
        #############################################################################################

        # self.labels 是一个 list, 长度为 self.dataset_size,
        # 每个元素是一个 np.array 表示每个 image 所有的 gt_box
        # 每一行分别表示 class_id, xmin, ymin, xmax, ymax
        if not self.labels:
            # 如果 self.labels 是 None or [], 要求返回下面的这些值是无理的
            if any([ret in returns for ret in
                    ['original_labels',
                     'processed_labels',
                     'encoded_labels',
                     'matched_anchors',
                     'evaluation_neutral']]):
                warnings.warn(
                    "Since no labels were given, none of 'original_labels', 'processed_labels', 'evaluation-neutral', "
                    "'encoded_labels', and 'matched_anchors' are possible returns, "
                    "but you set `returns = {}`. The impossible returns will be `None`.".format(returns))
        elif label_encoder is None:
            if any([ret in returns for ret in ['encoded_labels', 'matched_anchors']]):
                warnings.warn(
                    "Since no label encoder was given, 'encoded_labels' and 'matched_anchors' aren't possible returns, " 
                    "but you set `returns = {}`. The impossible returns will be `None`.".format(returns))
        elif not isinstance(label_encoder, SSDInputEncoder):
            if 'matched_anchors' in returns:
                warnings.warn(
                    "`label_encoder` is not an `SSDInputEncoder` object, "
                    "therefore 'matched_anchors' is not a possible return, "
                    "but you set `returns = {}`. The impossible returns will be `None`.".format(returns))

        #############################################################################################
        # Do a few preparatory things like maybe shuffling the dataset initially.
        #############################################################################################

        if shuffle:
            objects_to_shuffle = [self.dataset_indices]
            if self.filenames:
                objects_to_shuffle.append(self.filenames)
            if self.labels:
                objects_to_shuffle.append(self.labels)
            if self.image_ids:
                objects_to_shuffle.append(self.image_ids)
            if self.eval_neutral:
                objects_to_shuffle.append(self.eval_neutral)
            # 同时按相同的顺序 shuffle objects_to_shuffle 的所有元素, Note datasets_indices 是 np.array 其他都是 list
            shuffled_objects = sklearn.utils.shuffle(*objects_to_shuffle)
            for i in range(len(objects_to_shuffle)):
                # 与 objects_to_shuffle[i] = shuffled_objects[i], 区别是 [:] 直接在原数组上修改值, 而不是把整个数组重新赋值
                objects_to_shuffle[i][:] = shuffled_objects[i]

        # Override the labels formats of all the transformations to make sure they are set correctly.
        if self.labels:
            for transform in transformations:
                transform.labels_format = self.labels_output_format

        #############################################################################################
        # Generate mini batches.
        #############################################################################################

        current = 0
        while True:
            batch_x, batch_y = [], []
            if current >= self.dataset_size:
                current = 0
                #########################################################################################
                # Maybe shuffle the dataset if a full pass over the dataset has finished.
                #########################################################################################
                if shuffle:
                    objects_to_shuffle = [self.dataset_indices]
                    if self.filenames:
                        objects_to_shuffle.append(self.filenames)
                    if self.labels:
                        objects_to_shuffle.append(self.labels)
                    if self.image_ids:
                        objects_to_shuffle.append(self.image_ids)
                    if self.eval_neutral:
                        objects_to_shuffle.append(self.eval_neutral)
                    shuffled_objects = sklearn.utils.shuffle(*objects_to_shuffle)
                    for i in range(len(objects_to_shuffle)):
                        objects_to_shuffle[i][:] = shuffled_objects[i]

            #########################################################################################
            # Get the images, (maybe) image IDs, (maybe) labels, etc. for this batch.
            #########################################################################################

            # We prioritize our options in the following order:
            # 1) If we have the images already loaded in memory, get them from there.
            # 2) Else, if we have an HDF5 dataset, get the images from there.
            # 3) Else, if we have neither of the above, we'll have to load the individual image files from disk.
            batch_indices = self.dataset_indices[current:current + batch_size]
            if self.images:
                for i in batch_indices:
                    batch_x.append(self.images[i])
                if self.filenames:
                    batch_filenames = self.filenames[current:current + batch_size]
                else:
                    batch_filenames = None
            elif self.hdf5_dataset is not None:
                for i in batch_indices:
                    batch_x.append(self.hdf5_dataset['images'][i].reshape(self.hdf5_dataset['image_shapes'][i]))
                if self.filenames:
                    batch_filenames = self.filenames[current:current + batch_size]
                else:
                    batch_filenames = None
            else:
                if not self.filenames:
                    raise ValueError('`self.filenames` must not be None or []')
                else:
                    batch_filenames = self.filenames[current:current + batch_size]
                    for filename in batch_filenames:
                        with Image.open(filename) as image:
                            batch_x.append(np.array(image, dtype=np.uint8))

            # Get the labels for this batch (if there are any).
            if self.labels:
                batch_y = deepcopy(self.labels[current:current + batch_size])
            else:
                batch_y = None

            if self.eval_neutral:
                batch_eval_neutral = self.eval_neutral[current:current + batch_size]
            else:
                batch_eval_neutral = None

            # Get the image IDs for this batch (if there are any).
            if self.image_ids:
                batch_image_ids = self.image_ids[current:current + batch_size]
            else:
                batch_image_ids = None

            if 'original_images' in returns:
                # The original, unaltered images
                batch_original_images = deepcopy(batch_x)
            else:
                batch_original_images = None
            if 'original_labels' in returns and batch_y:
                # The original, unaltered labels
                batch_original_labels = deepcopy(batch_y)
            else:
                batch_original_labels = None

            current += batch_size

            #########################################################################################
            # Maybe perform image transformations.
            #########################################################################################
            # In case we need to remove any images from the batch, store their indices in this list.
            batch_items_to_remove = []
            batch_inverse_transforms = []

            for i in range(len(batch_x)):
                #########################################################################################
                # Check for if there is any gt box of this batch item.
                #########################################################################################
                if self.labels:
                    # Convert the labels for this image to an array (in case they aren't already).
                    batch_y[i] = np.array(batch_y[i])
                    # If this image has no ground truth boxes, maybe we don't want to keep it in the batch.
                    if (batch_y[i].size == 0) and not keep_images_without_gt:
                        batch_items_to_remove.append(i)
                        batch_inverse_transforms.append([])
                        continue

                #########################################################################################
                # Check for if batch item is valid after transformation
                #########################################################################################
                # Apply any image transformations we may have received.
                if transformations:
                    inverse_transforms = []
                    for transform in transformations:
                        if self.labels:
                            if ('inverse_transform' in returns) and (
                                    'return_inverter' in inspect.signature(transform).parameters):
                                batch_x[i], batch_y[i], inverse_transform = transform(batch_x[i], batch_y[i],
                                                                                      return_inverter=True)
                                inverse_transforms.append(inverse_transform)
                            else:
                                batch_x[i], batch_y[i] = transform(batch_x[i], batch_y[i])
                        else:
                            if ('inverse_transform' in returns) and (
                                    'return_inverter' in inspect.signature(transform).parameters):
                                batch_x[i], inverse_transform = transform(batch_x[i], return_inverter=True)
                                inverse_transforms.append(inverse_transform)
                            else:
                                batch_x[i] = transform(batch_x[i])

                        # In case the transform failed to produce an output image, which is possible for some random
                        # transforms. 究竟什么情况下才会发生这种情况?
                        if batch_x[i] is None:
                            batch_items_to_remove.append(i)
                            batch_inverse_transforms.append([])
                            # continue
                            # Adam
                            break
                    # transform 需要按照与原来相反的顺序存放
                    batch_inverse_transforms.append(inverse_transforms[::-1])

                #########################################################################################
                # Check for degenerate boxes in this batch item.
                #########################################################################################
                if self.labels:
                    xmin = self.labels_output_format.index('xmin')
                    ymin = self.labels_output_format.index('ymin')
                    xmax = self.labels_output_format.index('xmax')
                    ymax = self.labels_output_format.index('ymax')
                    if np.any(batch_y[i][:, xmax] - batch_y[i][:, xmin] <= 0) or np.any(
                            batch_y[i][:, ymax] - batch_y[i][:, ymin] <= 0):
                        if degenerate_box_handling == 'warn':
                            warnings.warn(
                                "Detected degenerate gt bounding boxes for batch item {} with bounding boxes {}, "
                                .format(i, batch_y[i]) +
                                "i.e. bounding boxes where x_max <= x_min and/or y_max <= y_min. " +
                                "This could mean that your dataset contains degenerate ground truth boxes, "
                                "or that any image transformations you may apply might result in degenerate gt boxes, "
                                "or that you are parsing the ground truth in the wrong coordinate format." 
                                "Degenerate ground truth bounding boxes may lead to NaN errors during the training.")
                        elif degenerate_box_handling == 'remove':
                            box_filter = BoxFilter(check_overlap=False,
                                                   check_min_area=False,
                                                   check_degenerate=True,
                                                   labels_format=self.labels_output_format)
                            batch_y[i] = box_filter(batch_y[i])
                            # 如果这个 image 的所有 gt_box 都被过滤掉, batch_y[i] 的 shape 为 (0, 5)
                            if (batch_y[i].size == 0) and not keep_images_without_gt:
                                batch_items_to_remove.append(i)

            #########################################################################################
            # Remove any items we might not want to keep from the batch.
            #########################################################################################
            if batch_items_to_remove:
                for j in sorted(batch_items_to_remove, reverse=True):
                    # This isn't efficient, but it hopefully shouldn't need to be done often anyway.
                    batch_x.pop(j)
                    batch_filenames.pop(j)
                    if batch_inverse_transforms:
                        batch_inverse_transforms.pop(j)
                    if self.labels:
                        batch_y.pop(j)
                    if self.image_ids:
                        batch_image_ids.pop(j)
                    if self.eval_neutral:
                        batch_eval_neutral.pop(j)
                    if batch_original_images:
                        batch_original_images.pop(j)
                    if batch_original_labels:
                        batch_original_labels.pop(j)

            #########################################################################################
            # CAUTION: Converting `batch_x` into an array will result in an empty batch if the images have varying sizes
            #          or varying numbers of channels. At this point, all images must have the same size and the same
            #          number of channels.
            batch_x = np.array(batch_x)
            if batch_x.size == 0:
                raise DegenerateBatchError(
                    "You produced an empty batch. This might be because the images in the batch vary " 
                    "in their size and/or number of channels. Note that after all transformations " 
                    "(if any were given) have been applied to all images in the batch, all images "
                    "must be homogeneous in size along all axes.")

            #########################################################################################
            # If we have a label encoder, encode our labels.
            #########################################################################################
            if (label_encoder is not None) and batch_y:
                if ('matched_anchors' in returns) and isinstance(label_encoder, SSDInputEncoder):
                    batch_y_encoded, batch_matched_anchors = label_encoder(batch_y, diagnostics=True)
                else:
                    batch_y_encoded = label_encoder(batch_y, diagnostics=False)
                    batch_matched_anchors = None
            else:
                batch_y_encoded = None
                batch_matched_anchors = None

            #########################################################################################
            # Compose the output.
            #########################################################################################
            ret = []
            if 'processed_images' in returns:
                # np.array
                ret.append(batch_x)
            if 'encoded_labels' in returns:
                # np.array
                ret.append(batch_y_encoded)
            if 'matched_anchors' in returns:
                # np.array
                ret.append(batch_matched_anchors)
            if 'processed_labels' in returns:
                # list
                ret.append(batch_y)
            if 'filenames' in returns:
                # list
                ret.append(batch_filenames)
            if 'image_ids' in returns:
                # list
                ret.append(batch_image_ids)
            if 'evaluation_neutral' in returns:
                # list
                ret.append(batch_eval_neutral)
            if 'inverse_transform' in returns:
                # list
                ret.append(batch_inverse_transforms)
            if 'original_images' in returns:
                # list
                ret.append(batch_original_images)
            if 'original_labels' in returns:
                # list
                ret.append(batch_original_labels)
            yield ret

    def save_dataset(self,
                     filenames_path='filenames.pkl',
                     labels_path=None,
                     image_ids_path=None,
                     eval_neutral_path=None):
        """
        Writes the current `filenames`, `labels`, and `image_ids` lists to the specified files.
        This is particularly useful for large datasets with annotations that are parsed from XML files, which can take
        quite long. If you'll be using the same dataset repeatedly, you don't want to have to parse the XML label
        files every time.

        Arguments:
            filenames_path (str): The path under which to save the filenames pickle.
            labels_path (str): The path under which to save the labels pickle.
            image_ids_path (str, optional): The path under which to save the image IDs pickle.
            eval_neutral_path (str, optional): The path under which to save the pickle for
                the evaluation-neutrality annotations.
        """
        if self.filenames:
            with open(filenames_path, 'wb') as f:
                pickle.dump(self.filenames, f)
        if self.labels and labels_path is not None:
            with open(labels_path, 'wb') as f:
                pickle.dump(self.labels, f)
        if self.image_ids and image_ids_path is not None:
            with open(image_ids_path, 'wb') as f:
                pickle.dump(self.image_ids, f)
        if self.eval_neutral and eval_neutral_path is not None:
            with open(eval_neutral_path, 'wb') as f:
                pickle.dump(self.eval_neutral, f)

    def get_dataset(self):
        """
        Returns:
            4-tuple containing lists and/or `None` for the filenames, labels, image IDs,
            and evaluation-neutrality annotations.
        """
        return self.filenames, self.labels, self.image_ids, self.eval_neutral

    def get_dataset_size(self):
        """
        Returns:
            The number of images in the dataset.
        """
        return self.dataset_size
