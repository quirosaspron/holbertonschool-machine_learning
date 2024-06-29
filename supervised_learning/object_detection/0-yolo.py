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
            return file.read().strip().split('n')
