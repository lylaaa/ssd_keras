from data_generator.object_detection_2d_data_generator import DataGenerator
import os.path as osp
from keras.models import load_model
from ssd_encoder_decoder.ssd_output_decoder import decode_detections
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
import cv2
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model = load_model('/home/adam/workspace/github/others/ssd_keras/ssd7_epoch-50_loss-1.7012_val_loss-1.9864.h5',
                   custom_objects={'AnchorBoxes': AnchorBoxes,
                                   'compute_loss': ssd_loss.compute_loss}
                   )
normalize_coords = True
# Height of the input images
img_height = 300
# Width of the input images
img_width = 480
images_dir = '/home/adam/.keras/datasets/udacity_self_driving_car/ssd_dataset'
# Ground truth
val_hdf5_path = osp.join(images_dir, 'ssd_val.h5')
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=val_hdf5_path)
predict_generator = val_dataset.generate(batch_size=1,
                                         shuffle=True,
                                         transformations=[],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'processed_labels',
                                                  'filenames'},
                                         keep_images_without_gt=False)
# 2: Generate samples
while True:
    batch_images, batch_labels, batch_filenames = next(predict_generator)
    # Which batch item to look at
    i = 0
    # print("Image:", batch_filenames[i])
    print("Ground truth boxes:\n")
    print(batch_labels[i])
    # 3: Make a prediction
    y_pred = model.predict(batch_images)
    # 4: Decode the raw prediction `y_pred`
    y_pred_decoded = decode_detections(y_pred,
                                       confidence_thresh=0.5,
                                       iou_threshold=0.45,
                                       top_k=200,
                                       normalize_coords=normalize_coords,
                                       img_height=img_height,
                                       img_width=img_width)

    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print(y_pred_decoded[i])
    # 5: Draw the predicted boxes onto the image
    classes = ['background', 'car', 'truck', 'pedestrian', 'bicyclist', 'traffic_light']
    image = batch_images[i]

    # Draw the ground truth boxes in green (omit the label for more clarity)
    for box in batch_labels[i]:
        xmin = box[1]
        ymin = box[2]
        xmax = box[3]
        ymax = box[4]
        class_id = int(box[0])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

    # Draw the predicted boxes in blue
    for box in y_pred_decoded[i]:
        xmin = int(round(box[-4]))
        ymin = int(round(box[-3]))
        xmax = int(round(box[-2]))
        ymax = int(round(box[-1]))
        class_id = int(box[0])
        cv2.putText(image, classes[class_id], (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', image)
    cv2.waitKey(0)
