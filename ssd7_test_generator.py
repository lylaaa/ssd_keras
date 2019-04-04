import os.path as osp
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from bounding_box_utils.bounding_box_utils import convert_coordinates
import numpy as np
import cv2

train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
images_dir = '/home/adam/.keras/datasets/udacity_self_driving_car/object-detection-crowdai-480-300'
# Ground truth
train_labels_filepath = osp.join(images_dir, 'train.csv')
train_dataset.parse_csv(images_dir=images_dir,
                        labels_filename=train_labels_filepath,
                        # This is the order of the first six columns in the CSV file that contains the labels for your
                        # dataset. If your labels are in XML format, maybe the XML parser will be helpful.
                        input_format=['image_name', 'xmin', 'ymin', 'xmax', 'ymax', 'class_id'],
                        include_classes='all')
class_names = ['car', 'truck', 'pedestrian']
data_augmentation_chain = DataAugmentationConstantInputSize(random_brightness=(-48, 48, 0.5),
                                                            random_contrast=(0.5, 1.8, 0.5),
                                                            random_saturation=(0.5, 1.8, 0.5),
                                                            random_hue=(18, 0.5),
                                                            random_flip=0.5,
                                                            random_translate=((0.03, 0.5), (0.03, 0.5), 0.5),
                                                            random_scale=(0.5, 2.0, 0.5),
                                                            n_trials_max=3,
                                                            # 这里 clip 的是 gt boxes
                                                            clip_boxes=True,
                                                            overlap_criterion_box_filter='area',
                                                            overlap_criterion_validator='area',
                                                            bounds_box_filter=(0.3, 1.0),
                                                            bounds_validator=(0.5, 1.0),
                                                            n_boxes_min=1,
                                                            background=(0, 0, 0))
batch_size = 4
img_height = 300
img_width = 480
n_classes = 3
scales = [0.08, 0.16, 0.32, 0.64, 0.96]
aspect_ratios = [0.5, 1.0, 2.0]
two_boxes_for_ar1 = True
steps = None
offsets = None
clip_boxes = False
variances = [1.0, 1.0, 1.0, 1.0]
normalize_coords = True

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
# [(img_height // 8, img_width // 8), (img_height // 16, img_width // 16), (img_height // 32, img_width // 32)
#  (img_height // 64, img_width // 64)]
predictor_sizes = [(150, 240), (75, 120), (37, 60), (18, 30)]
ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_global=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    # 这里 clip 的是 anchor boxes
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.3,
                                    normalize_coords=normalize_coords,
                                    coords='centroids',
                                    )

# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.
train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=(data_augmentation_chain,),
                                         label_encoder=ssd_input_encoder,
                                         returns=('processed_images',
                                                  'encoded_labels', 'original_images', 'original_labels'),
                                         keep_images_without_gt=False)


def preview_gt_boxes():
    for _, _, original_images, original_labels in train_generator:
        for i in range(len(original_images)):
            image = original_images[i]
            gt_boxes = original_labels[i]
            for gt_box in gt_boxes:
                cv2.putText(image, class_names[gt_box[0] - 1], (gt_box[1], gt_box[2]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                cv2.rectangle(image, (gt_box[1], gt_box[2]), (gt_box[3], gt_box[4]), (0, 255, 0), 2)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image', image)
            cv2.waitKey(0)


def preview_anchor_boxes():
    for _, _, original_images, original_labels in train_generator:
        for i in range(len(original_images)):
            original_image = original_images[i]
            original_label = original_labels[i]
            anchor_boxes = batch_item_y[:, -8:-4]
            anchor_boxes = convert_coordinates(anchor_boxes,
                                               start_index=0,
                                               conversion='centroids2corners',
                                               border_pixels='half')
            anchor_boxes[:, [0, 2]] *= img_width
            anchor_boxes[:, [1, 3]] *= img_height
            anchor_boxes = np.round(anchor_boxes).astype('int')
            print(anchor_boxes[0])
            image = batch_item_x.astype('int8')
            for anchor_box in anchor_boxes:
                cv2.rectangle(image, (anchor_box[0], anchor_box[1]), (anchor_box[2], anchor_box[3]), (0, 255, 0), 2)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image', image)
            cv2.waitKey(0)
            pass


preview_gt_boxes()
