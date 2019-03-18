import os.path as osp
import csv
import cv2
import pandas as pd
import numpy as np
import logging
import sys
import random

logger = logging.getLogger('irf_sh')
logger.setLevel(logging.DEBUG)  # default log level
format = logging.Formatter("%(asctime)s %(name)-8s %(levelname)-8s %(lineno)-4d %(message)s")  # output format
sh = logging.StreamHandler(stream=sys.stdout)  # output to standard output
sh.setFormatter(format)
logger.addHandler(sh)


def split_labels_csv(images_dir):
    labels_csv_path = osp.join(images_dir, 'labels_crowdai.csv')
    df = pd.read_csv(labels_csv_path)
    class_names = df['Label'].unique().tolist()
    csv_reader = csv.reader(open(labels_csv_path, 'r'), delimiter=',')
    # 忽略第一行 header
    next(csv_reader)
    # image file name 作为 key, [[class_id,xmin,ymin,xmax,ymax],...] 作为 value
    annotations = {}
    for row in csv_reader:
        xmin = row[0]
        ymin = row[1]
        xmax = row[2]
        ymax = row[3]
        image_filename = row[4]
        class_name = row[5]
        # because id=0 reversed for background
        class_id = class_names.index(class_name) + 1
        if image_filename not in annotations:
            annotations[image_filename] = []
        annotations[image_filename].append([image_filename, xmin, ymin, xmax, ymax, class_id])
    # 9218
    num_images = len(annotations)
    # 6452
    num_train_images = int(num_images * 0.7)
    # 2766
    num_val_images = num_images - num_train_images
    logger.debug(
        'num_images={}, num_train_images={}, num_val_image={}'.format(num_images, num_train_images, num_val_images))

    train_csv_obj = open(osp.join(images_dir, 'train.csv'), 'w')
    train_csv_writer = csv.writer(train_csv_obj, delimiter=',')
    val_csv_obj = open(osp.join(images_dir, 'val.csv'), 'w')
    val_csv_writer = csv.writer(val_csv_obj, delimiter=',')
    image_filenames = list(annotations.keys())
    random.shuffle(image_filenames)
    for image_filename in image_filenames[:num_train_images]:
        for row in annotations[image_filename]:
            train_csv_writer.writerow(row)
    for image_filename in image_filenames[num_train_images:]:
        for row in annotations[image_filename]:
            val_csv_writer.writerow(row)


# split_labels_csv('/home/adam/.keras/datasets/udacity_self_driving_car/object-detection-crowdai')


def test_flip(image_path):
    image = cv2.imread(image_path)
    h_flipped_image1 = image[:, ::-1]
    h_flipped_image2 = cv2.flip(image, 1)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', image)
    cv2.namedWindow('h_flipped_image1', cv2.WINDOW_NORMAL)
    cv2.imshow('h_flipped_image1', h_flipped_image1)
    cv2.namedWindow('h_flipped_image2', cv2.WINDOW_NORMAL)
    cv2.imshow('h_flipped_image2', h_flipped_image2)
    cv2.waitKey(0)


# test_flip('/home/adam/.keras/datasets/udacity_self_driving_car/object-detection-crowdai/1479506172970816105.jpg')

def test_scale(image_path):
    image = cv2.imread(image_path)
    src_image = image.copy()
    img_height, img_width = image.shape[:2]
    # Compute the rotation matrix.
    rotation_matrix = cv2.getRotationMatrix2D(center=(img_width / 2, img_height / 2),
                                              angle=0,
                                              scale=0.8)
    image = cv2.warpAffine(image,
                           M=rotation_matrix,
                           dsize=(img_width, img_height),
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(0, 0, 0))
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', image)
    gt_box = (386, 571, 509, 661)
    top_left = np.array([gt_box[0], gt_box[1], 1])
    bottom_right = np.array([gt_box[2], gt_box[3], 1])
    new_top_left = np.dot(rotation_matrix, top_left)
    new_bottom_right = np.dot(rotation_matrix, bottom_right)
    new_top_left = np.round(new_top_left, decimals=0).astype(np.int).tolist()
    new_bottom_right = np.round(new_bottom_right, decimals=0).astype(np.int).tolist()
    # cv2.rectangle(src_image, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 255, 0), 2)
    cv2.rectangle(image, tuple(new_top_left), tuple(new_bottom_right), (0, 255, 0), 2)
    cv2.namedWindow('scaled_image', cv2.WINDOW_NORMAL)
    cv2.imshow('scaled_image', image)
    cv2.waitKey(0)


# test_scale('/home/adam/.keras/datasets/udacity_self_driving_car/object-detection-crowdai/1479506172970816105.jpg')
def test_rotate(image_path):
    image = cv2.imread(image_path)
    src_image = image.copy()
    img_height, img_width = image.shape[:2]
    # Compute the rotation matrix.
    rotation_matrix = cv2.getRotationMatrix2D(center=(img_width / 2, img_height / 2),
                                              angle=90,
                                              scale=1)
    cos_angle = np.abs(rotation_matrix[0, 0])
    sin_angle = np.abs(rotation_matrix[0, 1])
    # Compute the new bounding dimensions of the image.
    img_width_new = int(img_height * sin_angle + img_width * cos_angle)
    img_height_new = int(img_height * cos_angle + img_width * sin_angle)
    # Adjust the rotation matrix to take into account the translation.
    rotation_matrix[1, 2] += (img_height_new - img_height) / 2
    rotation_matrix[0, 2] += (img_width_new - img_width) / 2
    image = cv2.warpAffine(image,
                           M=rotation_matrix,
                           dsize=(img_width_new, img_height_new),
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(0, 0, 0))
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', image)
    cv2.waitKey(0)


test_rotate('/home/adam/.keras/datasets/udacity_self_driving_car/object-detection-crowdai/1479506172970816105.jpg')
