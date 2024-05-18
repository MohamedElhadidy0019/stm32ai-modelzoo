import os
import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from copy import deepcopy
import cv2
import PIL

#class_names: [Australian-King-Parrot,Crimson-Rosella,Sugar-Glider,Common-Brushtail-Possum,Musk-Lorikeet,Australian-Owlet-nightjar,Southern-Boobook,Gang-gang-Cockatoo,Barn-Owl,Rainbow-Lorikeet,Common-Ringtail-Possum,Yellow-tailed-Black-Cockatoo,Laughing-Kookaburr,Australian-Wood-Duck,Powerful-Owl]
#hcange this to array of strings
class_names = ['Australian-King-Parrot','Crimson-Rosella','Sugar-Glider','Common-Brushtail-Possum','Musk-Lorikeet','Australian-Owlet-nightjar','Southern-Boobook','Gang-gang-Cockatoo','Barn-Owl','Rainbow-Lorikeet','Common-Ringtail-Possum','Yellow-tailed-Black-Cockatoo','Laughing-Kookaburr','Australian-Wood-Duck','Powerful-Owl']

def interval_overlap(interval_a, interval_b):
    """
    Find overlap between two intervals
    Arguments:
        interval_a: [x1, x2]
        interval_b: [x3, x4]
    Returns:
        overlapped distance
    """
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou(box1, box2):
    """
    Find IoU between two boxes
    Arguments:
        box1 = [xmin, ymin, xmax, ymax]
        box2 = [xmin, ymin, xmax, ymax]
    Returns:
        iou similarity
    """
    intersect_w = interval_overlap([box1[0], box1[2]], [box2[0], box2[2]])
    intersect_h = interval_overlap([box1[1], box1[3]], [box2[1], box2[3]])
    intersect = intersect_w * intersect_h
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    union = w1 * h1 + w2 * h2 - intersect
    return float(intersect) / union

def load_and_pre_process_image(image_path, image_size=(256, 256)):
    channels=3
    data = tf.io.read_file(image_path)

    image = tf.io.decode_image(data, channels=channels, expand_animations=False)
    image = tf.image.resize(image, [image_size[1], image_size[0]], method='nearest')

    scale = 127.5
    offset = -1
    image = tf.cast(image, tf.float32) * scale + offset

    return image.numpy()
    

def decode_predictions(predictions: np.ndarray, normalize: bool = True, org_img_height: int = None,
                       org_img_width: int = None) -> np.ndarray:
    """
    Retrieve object bounding box coordinates from predicted offsets and anchor box coordinates.

    Args:
        predictions (np.ndarray): The output of an SSD-based human detection model, a tensor of [None, #boxes, 1+n_classes+4+4].
        normalize (bool): Whether the coordinates are normalized or not.
        org_img_height (int): The original image height, used if normalize=True.
        org_img_width (int): The original image width, used if normalize=True.

    Returns:
        predictions_decoded_raw (np.ndarray): The object bounding boxes and categories, a tensor of [None, #boxes, 1+n_classes+4].
    """
    predictions_decoded_raw = np.copy(predictions[:, :, :-4])

    # Unnormalize the offsets with the height and width of anchor boxes
    predictions_decoded_raw[:, :, [-4, -2]] *= np.expand_dims(predictions[:, :, -2] - predictions[:, :, -4], axis=-1)
    predictions_decoded_raw[:, :, [-3, -1]] *= np.expand_dims(predictions[:, :, -1] - predictions[:, :, -3], axis=-1)
    predictions_decoded_raw[:, :, -4:] += predictions[:, :, -4:]

    if normalize:
        predictions_decoded_raw[:, :, [-4, -2]] *= org_img_width
        predictions_decoded_raw[:, :, [-3, -1]] *= org_img_height

    return predictions_decoded_raw



def do_nms(preds_decoded, nms_thresh=0.45, confidence_thresh=0.5):
    """
    Non-maximum suppression, removing overlapped bounding boxes based on IoU metric and keeping bounding boxes with the highest score
    Arguments:
        preds_decoded: return of decode_predictions function, a tensor of [None, #boxes, 1+n_classes+4]
        nms_thresh: IoU threshold to remove overlapped bounding boxes, a float between 0 and 1
        confidence_thresh: minimum score to keep bounding boxes, a float between 0 and 1
    Returns:
        final_preds: detection results after non-maximum suppression
    """
    n_classes_bg = int(preds_decoded.shape[2]) - 4

    final_preds = dict()

    for b_item in preds_decoded:  # loop for each batch item
        for c in range(1, n_classes_bg):  # loop for each object category
            single_class = b_item[:,
                           [c, -4, -3, -2, -1]]  # retrieve predictions [score, xmin, ymin, xmax, ymax] for category c
            threshold_met = single_class[
                single_class[:, 0] > confidence_thresh]  # filter predictions with minimum confidence score threshold
            if threshold_met.shape[0] > 0:
                # sort confidence score in descending order
                sorted_indices = np.argsort([-elm[0] for elm in threshold_met])
                for i in range(len(sorted_indices)):  # loop for bounding boxes in order of highest score
                    index_i = sorted_indices[i]
                    if threshold_met[index_i, 0] == 0:  # verify if this box was processed
                        continue
                    box_i = threshold_met[index_i, -4:]  # get [xmin, ymin, xmax, ymax] of box_i
                    for j in range(i + 1, len(sorted_indices)):
                        index_j = sorted_indices[j]
                        if threshold_met[index_j, 0] == 0:  # verify if this box was processed
                            continue
                        box_j = threshold_met[index_j, -4:]  # get [xmin, ymin, xmax, ymax] of box_j
                        if bbox_iou(box_i,
                                    box_j) >= nms_thresh:  # check if two boxes are overlapped based on IoU metric
                            threshold_met[index_j, 0] = 0  # if Yes, remove bounding box with smaller confidence score
                threshold_met = threshold_met[threshold_met[:, 0] > 0]
            final_preds[c] = threshold_met  # detection results after non-maximum suppression of object category c
    
    return final_preds



def postprocess_predictions(predictions: Dict[str, List[Tuple[float, float, float, float, float]]],
                            image_size: int = None,
                            nms_thresh: float = 0.5,
                            confidence_thresh: float = 0.5) -> Dict[str, List[Tuple[float, float, float, float]]]:
    """
    Postprocesses the predictions to filter out weak and overlapping bounding boxes.

    Args:
        predictions: A dictionary of predictions for each class.
        height: The height of the original image.
        width: The width of the original image.
        nms_thresh: The IoU threshold for non-maximum suppression.
        confidence_thresh: The confidence threshold for filtering out weak predictions.

    Returns:
        A dictionary of filtered predictions for each class.
    """

    predicted_scores, predicted_boxes, predicted_anchors = predictions
    predictions = np.concatenate([predicted_scores, predicted_boxes, predicted_anchors], axis=2)

    preds_decoded = decode_predictions(predictions, normalize=True, 
                                       org_img_height=image_size[1], org_img_width=image_size[0])
    final_preds = do_nms(preds_decoded, nms_thresh=float(nms_thresh),
                         confidence_thresh=float(confidence_thresh))
    return final_preds


def corners_to_center_box_coords(x_min, x_max, y_min, y_max, image_size=None, relative=None):
    """
    Converts a predicted bounding box from (class_id, x_min, y_min, x_max, y_max) format 
    to (class_id, x_center, y_center, w, h) or (class_id, x_r, y_r, w_r, h_r) format.

    Args:
        image (np.ndarray): The input image.
        box (List[Union[int, float]]): The predicted bounding box in (class_id, x_min, y_min, x_max, y_max) format.
        # the box input size is in full size not relative
        relative (bool): Whether to return the bounding box coordinates as relative to the image size.

    Returns:
        The predicted bounding box in (class_id, x_center, y_center, w, h) or (class_id, x_r, y_r, w_r, h_r) format.
    """
    width, height = image_size

    w = x_max - x_min
    h = y_max - y_min
    x_center = x_min + w / 2
    y_center = y_min + h / 2

    if relative:
        x_center /= width
        y_center /= height
        w /= width
        h /= height

    return x_center, y_center, w, h

def predictions_to_bbox(predictions, image_size):
    """
    Convert predictions to bounding boxes
    Args:
        predictions: A dictionary of predictions for each class.
        image_size: The size of the original image.
    Returns:
        A dictionary of bounding boxes for each class.
    """
    bboxes = [] # class, x_center, y_center, width, height

    for predicted_class, detection_boxes in predictions.items():
        for bbox in detection_boxes:
            score, xmi, ymi, xma, yma = bbox
            x_center, y_center, w, h = corners_to_center_box_coords(xmi, xma, ymi, yma, image_size)
            bboxes.append([predicted_class, x_center, y_center, w, h])
    return bboxes



'''
write a function that takes an image, and bbox list of format [class, x_center, y_center, w, h] normalised
and vis it using matplotlib
'''
def vis_bbox(image, bboxes):
    fig, ax = plt.subplots()
    ax.imshow(image)
    cls = 0
    for bbox in bboxes:
        cls = bbox[0]
        x_center, y_center, w, h = bbox[1:]
        xmi = x_center - w/2
        ymi = y_center - h/2
        xma = x_center + w/2
        yma = y_center + h/2
        rect = plt.Rectangle((xmi, ymi), w, h, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(xmi, ymi, f'{bbox[0]}', fontsize=10, color='red')
    name = class_names[cls-1]
    ax.set_title(name + ' ' + str(cls))
    plt.show()

    pass


def main():
    image_size = (256, 256)
    nms_thresh = 0.5
    confidence_thresh = 0.4

    image_path = '/home/mohamed/repos/bird_detection/bird_detection/split_darknet/val_mini/images/42875.jpg'
    model_path = '/home/mohamed/repos/bird_detection/bird_detection/stm32ai-modelzoo/object_detection/src/quantized_model.tflite'

    interpreter = tf.lite.Interpreter(model_path)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()

    image = load_and_pre_process_image(image_path, image_size)
    image = np.expand_dims(image, axis=0)

    # 8-bit quantization approximates floating point values
    # using the following formula:
    #     float_value = (int8_value - zero_points) * scale
    # so to get the int8 values:
    #     int8_value = float_value/scale + zero_points
    tf_scale = input_details['quantization'][0] # which is 127,5
    tf_zero_points = input_details['quantization'][1] # which is 0
    image = np.round(image / tf_scale + tf_zero_points)
    input_dtype = input_details['dtype']
    image = image.astype(dtype=input_dtype)
    image = np.clip(image, np.iinfo(input_dtype).min, np.iinfo(input_dtype).max)                 

    # Make a prediction for the image to get detection boxes
    interpreter.set_tensor(input_index, image)
    interpreter.invoke()

    predicted_scores = interpreter.get_tensor(output_index[0]["index"])
    predicted_boxes = interpreter.get_tensor(output_index[1]["index"])
    predicted_anchors = interpreter.get_tensor(output_index[2]["index"])

    raw_predictions = [predicted_scores, predicted_boxes, predicted_anchors]

    predictions = postprocess_predictions(
                                raw_predictions,
                                image_size=image_size,
                                nms_thresh=nms_thresh,
                                confidence_thresh=confidence_thresh)
    
    bboxes = predictions_to_bbox(predictions, image_size)

    # read using pil
    image_pil = PIL.Image.open(image_path)
    image_pil = image_pil.resize(image_size)
    vis_bbox(image_pil, bboxes)


    pass



if __name__ == '__main__':
    main()