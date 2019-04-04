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

model = ssd_300(image_size=(image_height, image_width, 3),
                n_classes=20,
                mode='inference',
                l2_regularization=0.0005,
                # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

# plot_model(model, 'ssd300.png', show_shapes=True)

# Load the trained weights into the model
weights_path = '/home/adam/workspace/github/xuannianz/ssd_keras/ssd300_pascal_07+12_102k.h5'
model.load_weights(weights_path, by_name=True)

# Compile the model so that Keras won't complain the next time you load it.
# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

############################################################
# 1.2: Load a trained model
############################################################

# model_path = '/home/adam/workspace/github/xuannianz/ssd_keras/ssd300_pascal_07+12_102k.h5'
# model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
#                                                'L2Normalization': L2Normalization,
#                                                'DecodeDetections': DecodeDetections,
#                                                'compute_loss': ssd_loss.compute_loss})

############################################################
# 2. Load some images
############################################################

# Store the images here.
# orig_images = []
# Store resized versions of the images here.
# input_images = []
# We'll only load one image in this example.
# image_path = '/home/adam/.keras/datasets/VOCdevkit/test/VOC2007/JPEGImages/000096.jpg'
# orig_images.append(imread(image_path))
# image_ = image.load_img(image_path, target_size=(image_height, image_width))
# image_ = image.img_to_array(image_)
# input_images.append(image_)
# input_images = np.array(input_images)

############################################################
# 3. Make predictions
############################################################

# y_pred = model.predict(input_images)
# Decode the raw prediction `y_pred`
# 如果 build model 且指定了 mode='inference', 不需要再 decode prediction
# 如果是 load_model 需要再 decode prediction
# y_pred_decoded = decode_detections(y_pred,
#                                    confidence_thresh=0.5,
#                                    iou_threshold=0.45,
#                                    top_k=200,
#                                    normalize_coords=True,
#                                    img_height=image_height,
#                                    img_width=image_width)
#
# np.set_printoptions(precision=2, suppress=True, linewidth=90)
# print("Predicted boxes:\n")
# print('   class   conf xmin   ymin   xmax   ymax')
# print(y_pred_decoded[0])

############################################################
# 4. Draw the predicted boxes onto the image
############################################################

classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
# image = orig_images[0][:, :, ::-1]

# Draw the predicted boxes in red
# for box in y_pred_decoded[0]:
#     xmin = int(round(box[-4] * orig_images[0].shape[1] / image_width))
#     ymin = int(round(box[-3] * orig_images[0].shape[0] / image_height))
#     xmax = int(round(box[-2] * orig_images[0].shape[1] / image_width))
#     ymax = int(round(box[-1] * orig_images[0].shape[0] / image_height))
#     class_id = int(box[0])
#     cv2.putText(image, classes[class_id], (xmin, ymin + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
#     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image', image)
# cv2.waitKey(0)

images_dir = '/home/adam/.keras/datasets/VOCdevkit'
# Ground truth
val_hdf5_path = osp.join(images_dir, '07_test.h5')
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=val_hdf5_path)
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=image_height, width=image_width)
predict_generator = val_dataset.generate(batch_size=1,
                                         shuffle=True,
                                         transformations=[convert_to_3_channels, resize],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'filenames',
                                                  'inverse_transform',
                                                  'original_images',
                                                  'original_labels'
                                                  },
                                         keep_images_without_gt=False)
# 2: Generate samples
while True:
    batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(
        predict_generator)
    # print("Image:", batch_filenames[0])
    print("Ground truth boxes:\n")
    print(batch_original_labels[0])
    # 3: Make a prediction
    y_pred = model.predict(batch_images)
    mode = 'inference'
    if mode == 'training':
        # 4: Decode the raw prediction `y_pred`
        y_pred_decoded = decode_detections(y_pred,
                                           confidence_thresh=0.5,
                                           iou_threshold=0.45,
                                           top_k=200,
                                           normalize_coords=True,
                                           img_height=image_height,
                                           img_width=image_width)
    else:
        y_pred_decoded = [y_pred[k][y_pred[k, :, 1] > 0.5] for k in range(y_pred.shape[0])]
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print(y_pred_decoded[0])
    image = batch_original_images[0][:, :, ::-1]
    image = image.copy()

    # Draw the ground truth boxes in green (omit the label for more clarity)
    for box in batch_original_labels[0]:
        xmin = box[1]
        ymin = box[2]
        xmax = box[3]
        ymax = box[4]
        class_id = int(box[0])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

    # Draw the predicted boxes in blue
    if len(y_pred_decoded[0]) > 0:
        y_pred_decoded = np.array(y_pred_decoded)
        y_pred = apply_inverse_transforms(y_pred_decoded[:, :, [0, 2, 3, 4, 5]], batch_inverse_transforms)
        for box in y_pred[0]:
            xmin = int(round(box[-4]))
            ymin = int(round(box[-3]))
            xmax = int(round(box[-2]))
            ymax = int(round(box[-1]))
            class_id = int(box[0])
            cv2.putText(image, classes[class_id], (xmin, ymin + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', image)
    cv2.waitKey(0)
