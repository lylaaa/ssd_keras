from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger, ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
import os.path as osp
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
import datetime
import matplotlib
import cv2

matplotlib.use('Agg')

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

# 1: Build the Keras model.

# Clear previous models from memory.
K.clear_session()
model = ssd_300(image_size=(img_height, img_width, img_channels),
                n_classes=n_classes,
                mode='training',
                l2_regularization=0.0005,
                scales=scales,
                aspect_ratios_per_layer=aspect_ratios,
                two_boxes_for_ar1=two_boxes_for_ar1,
                steps=steps,
                offsets=offsets,
                clip_boxes=clip_boxes,
                variances=variances,
                normalize_coords=normalize_coords,
                subtract_mean=mean_color,
                swap_channels=swap_channels,
                # return_predictor_sizes=True
                )

# 2: Load some weights into the model.
# TODO: Set the path to the weights you want to load.
weights_path = 'VGG_ILSVRC_16_layers_fc_reduced.h5'
model.load_weights(weights_path, by_name=True)

# 3: Instantiate an optimizer and the SSD loss function and compile the model.
#    If you want to follow the original Caffe implementation, use the preset SGD
#    optimizer, otherwise I'd recommend the commented-out Adam optimizer.
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

############################################################################
# Set up the data generators for the training
############################################################################

# Instantiate two `DataGenerator` objects: One for training, one for validation.
# Optional: If you have enough memory, consider loading the images into memory for the reasons explained above.
DATASET_DIR = '/home/adam/.keras/datasets/VOCdevkit'
train_hdf5_path = osp.join(DATASET_DIR, '07+12_trainval.h5')
val_hdf5_path = osp.join(DATASET_DIR, '07_test.h5')

train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=train_hdf5_path)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=val_hdf5_path)

# Parse the image and label lists for the training and validation datasets. This can take a while.

# The directories that contain the images.
# VOC_2007_trainval_images_dir = osp.join(DATASET_DIR, 'trainval/VOC2007/JPEGImages')
# VOC_2012_trainval_images_dir = osp.join(DATASET_DIR, 'trainval/VOC2012/JPEGImages')
# VOC_2007_test_images_dir = osp.join(DATASET_DIR, 'test/VOC2007/JPEGImages')

# The directories that contain the annotations.
# VOC_2007_trainval_annotations_dir = osp.join(DATASET_DIR, 'trainval/VOC2007/Annotations')
# VOC_2012_trainval_annotations_dir = osp.join(DATASET_DIR, 'trainval/VOC2012/Annotations')
# VOC_2007_test_annotations_dir = osp.join(DATASET_DIR, 'test/VOC2007/Annotations')

# The paths to the image sets.
# VOC_2007_trainval_image_set_filename = osp.join(DATASET_DIR, 'trainval/VOC2007/ImageSets/Main/trainval.txt')
# VOC_2012_trainval_image_set_filename = osp.join(DATASET_DIR, 'trainval/VOC2012/ImageSets/Main/trainval.txt')
# VOC_2007_test_image_set_filename = osp.join(DATASET_DIR, 'test/VOC2007/ImageSets/Main/test.txt')

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
#
# train_dataset.parse_xml(images_dirs=[VOC_2007_trainval_images_dir,
#                                      VOC_2012_trainval_images_dir],
#                         image_set_filenames=[VOC_2007_trainval_image_set_filename,
#                                              VOC_2012_trainval_image_set_filename],
#                         annotations_dirs=[VOC_2007_trainval_annotations_dir,
#                                           VOC_2012_trainval_annotations_dir],
#                         classes=classes,
#                         include_classes='all',
#                         exclude_truncated=False,
#                         exclude_difficult=False,
#                         ret=False)
#
# val_dataset.parse_xml(images_dirs=[VOC_2007_test_images_dir],
#                       image_set_filenames=[VOC_2007_test_image_set_filename],
#                       annotations_dirs=[VOC_2007_test_annotations_dir],
#                       classes=classes,
#                       include_classes='all',
#                       exclude_truncated=False,
#                       exclude_difficult=True,
#                       ret=False)

# Optional: Convert the dataset into an HDF5 dataset. This will require more disk space, but will speed up the training.
# Doing this is not relevant in case you activated the `load_images_into_memory` option in the constructor,
# because in that cas the images are in memory already anyway.
# If you don't want to create HDF5 datasets, comment out the subsequent two function calls.
# if not osp.exists(train_hdf5_path):
#     train_dataset.create_hdf5_dataset(file_path=train_hdf5_path,
#                                       resize=False,
#                                       variable_image_size=True,
#                                       verbose=True)
# if not osp.exists(val_hdf5_path):
#     val_dataset.create_hdf5_dataset(file_path=val_hdf5_path,
#                                     resize=False,
#                                     variable_image_size=True,
#                                     verbose=True)

# Change the batch size if you like, or if you run into GPU memory issues.
batch_size = 32

# Set the image transformations for pre-processing and data augmentation options for the training generator
ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                            img_width=img_width,
                                            background=mean_color)

# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]

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

# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         # transformations=[ssd_data_augmentation],
                                         transformations=[convert_to_3_channels, resize],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  # 'processed_labels',
                                                  'encoded_labels',
                                                  # 'inverse_transform'
                                                  },
                                         keep_images_without_gt=False)

# test generator
# colors = [np.random.randint(0, 256, 3).tolist() for i in range(len(classes))]
# for batch_processed_images, batch_processed_labels in train_generator:
#     batch_size = batch_processed_images.shape[0]
#     for i in range(batch_size):
#         image = batch_processed_images[i]
#         processed_labels = batch_processed_labels[i]
#         for processed_label in processed_labels:
#             class_id = int(processed_label[0])
#             class_name = classes[class_id]
#             xmin = processed_label[1]
#             ymin = processed_label[2]
#             xmax = processed_label[3]
#             ymax = processed_label[4]
#             label = '{}'.format(class_name)
#             color = colors[class_id - 1]
#             # ret[0] 表示包围 text 的矩形框的 width
#             # ret[1] 表示包围 text 的矩形框的 height
#             # baseline 表示的 text 最底下一个像素到文本 baseline 的距离
#             # 文本 baseline 参考 https://blog.csdn.net/u010970514/article/details/84075776
#             ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#             cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
#             cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
#             cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
#         cv2.imshow('image', image)
#         cv2.waitKey(0)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[convert_to_3_channels,
                                                      resize],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size = val_dataset.get_dataset_size()
print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))


# Define a learning rate schedule.
def lr_schedule(epoch):
    if epoch < 60:
        return 0.001
    elif epoch < 80:
        return 0.0001
    else:
        return 0.00001


# Define model callbacks.
model_checkpoint = ModelCheckpoint(filepath='ssd300_pascal_07+12_{epoch:02d}_{loss:.4f}_{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)

csv_logger = CSVLogger(filename='ssd300_pascal_07+12_training_log.csv',
                       separator=',',
                       append=True)

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6, verbose=1)

terminate_on_nan = TerminateOnNaN()
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

callbacks = [model_checkpoint,
             csv_logger,
             early_stopping,
             # learning_rate_scheduler,
             reduce_lr,
             terminate_on_nan]

# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch = 0
final_epoch = 120
steps_per_epoch = 1000

H = model.fit_generator(generator=train_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=final_epoch,
                        callbacks=callbacks,
                        validation_data=val_generator,
                        validation_steps=ceil(val_dataset_size / batch_size),
                        initial_epoch=initial_epoch)
N = len(H.history['loss'])
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Train and Val Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("plot-{}.jpg".format(str(datetime.date.today()).replace('-', '')))
