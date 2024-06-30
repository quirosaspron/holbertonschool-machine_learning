#!/usr/bin/env python3
"""Uses the Yolo v3 algorithm to perform object detection"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """Yolo class"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Yolo initializer"""
        self.model = K.models.load_model(model_path)
        self.class_names = self.load_list(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def load_list(self, file_path):
        """Loads a .txt file into a list"""
        with open(file_path, 'r') as file:
            return file.read().strip().split('\n')

    def process_outputs(self, outputs, image_size):
        """
        Process the outputs from the cnn
        outputs: List containing (grid_height,
        grid_width, anchor_boxes, 4 + 1 + classes)
        of the output predictions
        """
        boxes = []
        box_confidence = []
        box_class_probs = []

        for i, output in enumerate(outputs):

            grid_height, grid_width, anchor_boxes, lastclasse = output.shape
            image_height, image_width = image_size[0], image_size[1]

            # Review this further
            grid_x, grid_y = np.meshgrid(np.arange(grid_width),
                                         np.arange(grid_height))
            grid_x = grid_x.reshape(1, grid_height, grid_width, 1)
            grid_y = grid_y.reshape(1, grid_height, grid_width, 1)

            # Access t_x, t_y, t_w, and t_h
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            center_x = (K.activations.sigmoid(t_x) + grid_x)\
                / grid_width * image_width
            center_y = (K.activations.sigmoid(t_y) + grid_y)\
                / grid_height * image_height
            width = self.anchors[i][:, 0] * np.exp(t_w)\
                / self.model.input.shape[1] * image_width
            height = self.anchors[i][:, 1] * np.exp(t_h)\
                / self.model.input.shape[2] * image_height

            # (x1, y1) top left corner (x2, y2) bottom right corner
            x1 = center_x - (width / 2)
            y1 = center_y - (height / 2)
            x2 = center_x + (width / 2)
            y2 = center_y + (height / 2)

            box = np.zeros((grid_height, grid_width, anchor_boxes, 4))
            box[:, :, :, 0] = x1
            box[:, :, :, 1] = y1
            box[:, :, :, 2] = x2
            box[:, :, :, 3] = y2
            boxes.append(box)

            confidence = 1 / (1 + np.exp(-output[:, :, :, 4]))
            confidence = confidence.reshape(grid_height, grid_width,
                                            anchor_boxes, 1)
            box_confidence.append(confidence)
            box_class_probs_array = K.activations.sigmoid(output[..., 5:])
            box_class_probs.append(box_class_probs_array.numpy())

        return boxes, box_confidence, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters the bounding boxes based on the confidence
        of the predicted classes and the treshold set
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            box_scores_array = box_confidences[i] * box_class_probs[i]
            # Only keeps the highest prob class
            box_classes_array = np.argmax(box_scores_array, axis=-1)
            # Confidence in the predicted class of the bounding box
            box_class_scores = np.max(box_scores_array, axis=-1)
            # Keeps the boxes whose confidence passes the treshold
            pos = np.where(box_class_scores >= self.class_t)
            filtered_boxes.append(boxes[i][pos])
            box_classes.append(box_classes_array[pos])
            box_scores.append(box_class_scores[pos])

        # Makes one big np array out of all the np array elements
        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)

        return filtered_boxes, box_classes, box_scores
