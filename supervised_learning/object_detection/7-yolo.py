#!/usr/bin/env python3
"""Uses the Yolo v3 algorithm to perform object detection"""
import tensorflow.keras as K
import numpy as np
import cv2
import os


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

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        From the filtered boxes it only keeps the ones
        with the highest confidence
        """
        # Lexicographical sort, sorts by descending order of box scores
        # and sorts with box clasess for ties
        indices = np.lexsort((-box_scores, box_classes))
        # Applies sorting to boxes
        filtered_boxes = filtered_boxes[indices]
        box_scores = box_scores[indices]
        box_classes = box_classes[indices]
        # Extracts the unique classes
        unique_classes = np.unique(box_classes)
        nms_boxes = []
        nms_classes = []
        nms_scores = []

        for cls in unique_classes:
            # Gets all the indices of a particular class
            cls_indices = np.where(box_classes == cls)[0]
            cls_boxes = filtered_boxes[cls_indices]
            cls_scores = box_scores[cls_indices]

            while len(cls_boxes) > 0:
                # Adds the box with the max score
                max_score_index = np.argmax(cls_scores)
                nms_boxes.append(cls_boxes[max_score_index])
                nms_classes.append(cls)
                nms_scores.append(cls_scores[max_score_index])

                if len(cls_boxes) == 1:
                    break

                # Removes the box we added from the remaining boxes
                cls_boxes = np.delete(cls_boxes, max_score_index, axis=0)
                cls_scores = np.delete(cls_scores, max_score_index)
                # We get the boxes whose iou passes the treshold
                ious = self._iou(nms_boxes[-1], cls_boxes)
                iou_indices = np.where(ious <= self.nms_t)[0]
                # We repeat the process but only with the boxes
                # that passed the iou treshold
                cls_boxes = cls_boxes[iou_indices]
                cls_scores = cls_scores[iou_indices]

        return (np.array(nms_boxes),
                np.array(nms_classes),
                np.array(nms_scores))

    def _iou(self, box1, box2):
        """
        Calculates the Intersection Over Union (IOU) of two bounding boxes
        """
        x1 = np.maximum(box1[0], box2[:, 0])
        y1 = np.maximum(box1[1], box2[:, 1])
        x2 = np.minimum(box1[2], box2[:, 2])
        y2 = np.minimum(box1[3], box2[:, 3])

        intersection_area = np.maximum((x2 - x1), 0) * np.maximum((y2 - y1), 0)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union_area = box1_area + box2_area - intersection_area
        iou = intresection_area / union_area
        return iou

    @staticmethod
    def load_images(folder_path):
        """
        Loads images from the folder path
        """
        images = []
        image_paths = []
        # iterates though all the files in the directory
        for file in os.listdir(folder_path):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, file)
                image = cv2.imread(img_path)
                if image is not None:
                    images.append(image)
                    image_paths.append(img_path)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Resizes (modifies height and width)
        and rescales (normalizing to range [0,1])
        images for input to darket.
        """
        pimages = []
        image_shapes = []

        for image in images:
            # Gets heigth, width and channel of the image
            h, w, c = image.shape
            image_shapes.append([h, w])

            input_h = self.model.input.shape[1]
            input_w = self.model.input.shape[2]
            # Resize the image to the expected input of the model
            resized_img = cv2.resize(image, dsize=(input_h, input_w),
                                     interpolation=cv2.INTER_CUBIC)

            # Rescales the pixel values to range [0,1]
            # RGB pixel values have [0, 255.0] range
            resized_img = resized_img / 255.0

            pimages.append(resized_img)

        # Converts list to ndarray
        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Display the image with boundary boxes, class names, and box scores
        """
        for i, box in enumerate(boxes):
            # Applies int to every element of box
            x1, y1, x2, y2 = map(int, box)
            class_name = self.class_names[box_classes[i]]
            score = box_scores[i]

            # Display boundary box
            # Image, top and bottom corners, red and thickness of 2
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Display the class name and score
            text = f'{class_name} {score:.2f}'
            cv2.putText(image, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        if key == ord('s'):
            # Creates a detections directory if it doesn't exist
            if not os.path.exists('detections'):
                os.makedirs('detections')

            save_path = os.path.join('detections', file_name)
            cv2.imwrite(save_path, image)

        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Predicts and displays all images
        on the specified folder path
        """
        predictions = []
        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)
        model_predictions = self.model.predict(pimages)

        # Loops through every image
        for idx in range(len(pimages)):
            output = [p[idx] for p in model_predictions]
            # Processes the output of darknet
            (boxes, box_confidences,
             box_class_probs) = self.process_outputs(output, image_shapes[idx])
            # Filters the boxes
            (filtered_boxes, box_classes,
             box_scores) = self.filter_boxes(boxes,
                                             box_confidences, box_class_probs)
            # Applies non max supression
            (box_predictions,predicted_box_classes,
             predicted_box_scores) = self.non_max_suppression(filtered_boxes,
                                                              box_classes,
                                                              box_scores)
            # Adds the predictions to the list
            predictions.append((box_predictions,
                               predicted_box_classes,
                               predicted_box_scores))
            # Displays the predicted image
            self.show_boxes(image=images[idx],
                    boxes=box_predictions,
                    box_classes=predicted_box_classes,
                    box_scores=predicted_box_scores,
                    file_name=image_paths[idx])

        return predictions, image_paths
