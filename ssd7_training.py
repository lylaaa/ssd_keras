from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras import backend as K
from math import ceil
from matplotlib import pyplot as plt
from models.keras_ssd7 import build_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
import os.path as osp
import numpy as np
import datetime
import matplotlib

matplotlib.use('Agg')

# Height of the input images
img_height = 300
# Width of the input images
img_width = 480
# Number of color channels of the input images
img_channels = 3
# Set this to your preference (maybe `None`).
# The current settings transform the input pixel values to the interval `[-127.5,127.5]`.
intensity_mean = 127.5
# Set this to your preference (maybe `None`).
# The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5
# Number of positive classes, e.g. car, truck, pedestrian, bicyclist, traffic light
n_classes = 5
# An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
scales = [0.08, 0.16, 0.32, 0.64, 0.96]
# The list of aspect ratios for the anchor boxes
aspect_ratios = [0.5, 1.0, 2.0]
# Whether or not you want to generate two anchor boxes for aspect ratio 1
two_boxes_for_ar1 = True
# In case you'd like to set the step sizes for the anchor box grids manually; not recommended
steps = None
# In case you'd like to set the offsets for the anchor box grids manually; not recommended
offsets = None
# Whether or not to clip the anchor boxes to lie entirely within the image boundaries
clip_boxes = False
# The list/tuple of variances by which the encoded target coordinates are scaled
variances = [1.0, 1.0, 1.0, 1.0]
# Whether or not the model is supposed to use coordinates relative to the image size
normalize_coords = True

# 1: Build the Keras model
K.clear_session()  # Clear previous models from memory.
model = build_model(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_global=aspect_ratios,
                    aspect_ratios_per_layer=None,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=intensity_mean,
                    divide_by_stddev=intensity_range)
model.summary()

# 2: Optional: Load some weights
# model.load_weights('./ssd7_weights.h5', by_name=True)

# 3: Instantiate an Adam optimizer and the SSD loss function and compile the model
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# 1: Instantiate two `DataGenerator` objects: One for training, one for validation.
# Optional: If you have enough memory, consider loading the images into memory for the reasons explained above.

images_dir = '/home/adam/.keras/datasets/udacity_self_driving_car/ssd_dataset'
train_hdf5_path = osp.join(images_dir, 'ssd_train.h5')
val_hdf5_path = osp.join(images_dir, 'ssd_val.h5')
train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=train_hdf5_path)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=val_hdf5_path)

# 2: Parse the image and label lists for the training and validation datasets.
# Set the paths to your dataset here.
# Ground truth
# train_labels_filepath = osp.join(images_dir, 'labels_train.csv')
# val_labels_filepath = osp.join(images_dir, 'labels_val.csv')
# train_dataset.parse_csv(images_dir=images_dir,
#                         labels_filename=train_labels_filepath,
#                         # This is the order of the first six columns in the CSV file that contains the labels for your
#                         # dataset. If your labels are in XML format, maybe the XML parser will be helpful.
#                         input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
#                         include_classes='all')
#
# val_dataset.parse_csv(images_dir=images_dir,
#                       labels_filename=val_labels_filepath,
#                       input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
#                       include_classes='all')

# Optional: Convert the dataset into an HDF5 dataset. This will require more disk space, but will speed up the training.
# Doing this is not relevant in case you activated the `load_images_into_memory` option in the constructor,
# because in that case the images are in memory already anyway.
# If you don't want to create HDF5 datasets, comment out the subsequent two function calls.

# if not osp.exists(train_hdf5_path):
#     train_dataset.create_hdf5_dataset(file_path=train_hdf5_path,
#                                       resize=False,
#                                       variable_image_size=True,
#                                       verbose=True)
# if not osp.exists(val_hdf5_path):
#     val_dataset.create_hdf5_dataset(file_path=osp.join(images_dir, 'ssd_val.h5'),
#                                     resize=False,
#                                     variable_image_size=True,
#                                     verbose=True)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

# 3: Set the batch size.
batch_size = 32

# 4: Define the image processing chain.
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

# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
# [(img_height // 8, img_width // 8), (img_height // 16, img_width // 16), (img_height // 32, img_width // 32)
#  (img_height // 64, img_width // 64)]
predictor_sizes = [model.get_layer('classes4').output_shape[1:3],
                   model.get_layer('classes5').output_shape[1:3],
                   model.get_layer('classes6').output_shape[1:3],
                   model.get_layer('classes7').output_shape[1:3]]
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
                                         returns=('processed_images', 'encoded_labels'),
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=(),
                                     label_encoder=ssd_input_encoder,
                                     returns=('processed_images', 'encoded_labels'),
                                     keep_images_without_gt=False)
# Define model callbacks.
# TODO: Set the filepath under which you want to save the weights.
model_checkpoint = ModelCheckpoint(filepath='ssd7_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)

csv_logger = CSVLogger(filename='ssd7_training_log.csv',
                       separator=',',
                       append=True)

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0.0,
                               patience=10,
                               verbose=1)

# FIXME: cooldown 应该是修改 lr 后的第几个 epoch 开始重新 monitor
reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.2,
                                         patience=8,
                                         verbose=1,
                                         min_delta=0.001,
                                         cooldown=0,
                                         min_lr=0.00001)

callbacks = [model_checkpoint,
             csv_logger,
             early_stopping,
             reduce_learning_rate]
# TODO: Set the epochs to train for.
# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch = 0
final_epoch = 50
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
