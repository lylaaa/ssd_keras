from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt
from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
import os
import os.path as osp
from keras.utils.vis_utils import plot_model
import cv2
from eval_utils.average_precision_evaluator import Evaluator

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Set the image size.
image_height = 300
image_width = 300

# Clear previous models from memory.
K.clear_session()

# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

############################################################
# 1.1: Build the model and load trained weights into it
############################################################

# model = ssd_300(image_size=(image_height, image_width, 3),
#                 n_classes=20,
#                 mode='inference',
#                 l2_regularization=0.0005,
#                 # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
#                 scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
#                 aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
#                                          [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
#                                          [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
#                                          [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
#                                          [1.0, 2.0, 0.5],
#                                          [1.0, 2.0, 0.5]],
#                 two_boxes_for_ar1=True,
#                 steps=[8, 16, 32, 64, 100, 300],
#                 offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
#                 clip_boxes=False,
#                 variances=[0.1, 0.1, 0.2, 0.2],
#                 normalize_coords=True,
#                 subtract_mean=[123, 117, 104],
#                 swap_channels=[2, 1, 0],
#                 confidence_thresh=0.5,
#                 iou_threshold=0.45,
#                 top_k=200,
#                 nms_max_output_size=400)

# plot_model(model, 'ssd300.png', show_shapes=True)

# Load the trained weights into the model
# weights_path = '/home/adam/workspace/github/xuannianz/ssd_keras/ssd300_pascal_07+12_102k.h5'
# model.load_weights(weights_path, by_name=True)

# Compile the model so that Keras won't complain the next time you load it.
# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

############################################################
# 1.2: Load a trained model
############################################################

model_path = '/home/adam/workspace/github/xuannianz/ssd_keras/ssd300_pascal_07+12_120_4.6513_4.4621.h5'
model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'DecodeDetections': DecodeDetections,
                                               'compute_loss': ssd_loss.compute_loss})

############################################################
# 2. Load some images
############################################################


############################################################
# 3. Make predictions
############################################################


############################################################
# 4. Draw the predicted boxes onto the image
############################################################

classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
num_classes = len(classes) - 1

images_dir = '/home/adam/.keras/datasets/VOCdevkit'
# Ground truth
test_hdf5_path = osp.join(images_dir, '07_test.h5')
test_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=test_hdf5_path)
model_mode = 'training'
evaluator = Evaluator(model=model,
                      n_classes=num_classes,
                      data_generator=test_dataset,
                      model_mode=model_mode)

results = evaluator(img_height=image_height,
                    img_width=image_width,
                    batch_size=4,
                    data_generator_mode='resize',
                    round_confidences=False,
                    matching_iou_threshold=0.5,
                    border_pixels='include',
                    sorting_algorithm='quicksort',
                    average_precision_mode='sample',
                    num_recall_points=11,
                    ignore_neutral_boxes=True,
                    return_precisions=True,
                    return_recalls=True,
                    return_average_precisions=True,
                    verbose=True)
mean_average_precision, average_precisions, precisions, recalls = results
for i in range(1, len(average_precisions)):
    print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
print()
print("{:<14}{:<6}{}".format('', 'mAP', round(mean_average_precision, 3)))
