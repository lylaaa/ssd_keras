import os
import os.path as osp
from data_generator.object_detection_2d_data_generator import DataGenerator
import cv2
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation

DATASET_DIR = '/home/adam/.keras/datasets/VOCdevkit'
train_hdf5_path = osp.join(DATASET_DIR, '07+12_trainval.h5')
val_hdf5_path = osp.join(DATASET_DIR, '07_test.h5')
batch_size = 32
# Height of the model input images
img_height = 300
# Width of the model input images
img_width = 300
# Number of color channels of the model input images
img_channels = 3
# The per-channel mean of the images in the dataset.
# Do not change this value if you're using any of the pre-trained weights.
# RGB (123.7, 116.8, 103.9)
mean_color = [123, 117, 104]
# The color channel order in the original SSD is BGR,
# so we'll have the model reverse the color channel order of the input images.
swap_channels = [2, 1, 0]
# Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
n_classes = 20
# The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
# The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
scales = scales_pascal
# The anchor box aspect ratios used in the original SSD300; the order matters
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]
two_boxes_for_ar1 = True
# The space between two adjacent anchor box center points for each predictor layer.
steps = [8, 16, 32, 64, 100, 300]
# The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the
# step size for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
# Whether or not to clip the anchor boxes to lie entirely within the image boundaries
clip_boxes = False
# The variances by which the encoded target coordinates are divided as in the original implementation
variances = [0.1, 0.1, 0.2, 0.2]
normalize_coords = True

predictor_sizes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                            img_width=img_width,
                                            background=mean_color)

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.5,
                                    normalize_coords=normalize_coords)

train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=train_hdf5_path)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=val_hdf5_path)

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=(ssd_data_augmentation,),
                                         label_encoder=ssd_input_encoder,
                                         returns=('processed_images',
                                                  'encoded_labels', 'original_images', 'original_labels'),
                                         keep_images_without_gt=False)

class_names = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')


def preview_gt_boxes():
    for _, _, original_images, original_labels in train_generator:
        for i in range(len(original_images)):
            image = original_images[i]
            gt_boxes = original_labels[i]
            for gt_box in gt_boxes:
                cv2.putText(image, class_names[gt_box[0]], (gt_box[1], gt_box[2]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                cv2.rectangle(image, (gt_box[1], gt_box[2]), (gt_box[3], gt_box[4]), (0, 255, 0), 2)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image', image)
            cv2.waitKey(0)


def preview_processed_images():
    for processed_images, encoded_labels in train_generator:
        for i in range(len(processed_images)):
            image = processed_images[i]
            encoded_label = encoded_labels[i]
            for gt_box in gt_boxes:
                cv2.putText(image, class_names[gt_box[0]], (gt_box[1], gt_box[2]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                cv2.rectangle(image, (gt_box[1], gt_box[2]), (gt_box[3], gt_box[4]), (0, 255, 0), 2)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image', image)
            cv2.waitKey(0)


preview_gt_boxes()
