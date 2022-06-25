#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import copy
import rospkg

class YoloDetection:

    def __init__(self, n_cores=1, confThreshold=0.5, nmsThreshold=0.6, inpWidth=416, use_gpu=False):
        self.n_cores = n_cores
        self.PATH_PERCEPTION_UTLITIES = rospkg.RosPack().get_path('perception_utilities')
        self.CFG_PATH = self.PATH_PERCEPTION_UTLITIES+'/resources/yolo/yolov3-tiny_obj.cfg'
        self.WEIGHTS_PATH = self.PATH_PERCEPTION_UTLITIES+'/resources/yolo/yolov3-tiny_obj_last.weights'
        self.CLASS_NAMES_PATH = self.PATH_PERCEPTION_UTLITIES+'/resources/yolo/obj.names'
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.inpWidth = inpWidth
        self.model = cv.dnn.readNetFromDarknet(self.CFG_PATH, self.WEIGHTS_PATH)
        if use_gpu == True:
            self.model.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            self.model.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        else:
            cv.setNumThreads(self.n_cores)
        self.model.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.model.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        self.output_layer_names = self.model.getLayerNames()
        #breakpoint()
        #self.output_layer_names = [self.output_layer_names[i[0] - 1] for i in self.model.getUnconnectedOutLayers()]
        self.output_layer_names = [self.output_layer_names[i - 1] for i in self.model.getUnconnectedOutLayers()]
        self.labels = open(self.CLASS_NAMES_PATH).read().strip().split("\n")

    def Detection(self, image):
        (W, H) = (None, None)
        if W is None or H is None:
            (H, W) = image.shape[:2]

        blob = cv.dnn.blobFromImage(image, 1 / 255.0, (self.inpWidth, self.inpWidth), swapRB=True, crop=False)
        self.model.setInput(blob)
        model_output = self.model.forward(self.output_layer_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in model_output:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confThreshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        outputs_wrapper = list()
        idxs = cv.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        #print("Confidences: ", confidences)
        #print("Scores: ", scores)
        #print("Class_id: ", class_id)
        #print("Class_ids: ", class_ids)
        #print("outputs_wrapper: ", outputs_wrapper)
        #print("Labels: ", self.labels)
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]
                outputs_wrapper.append([x, y, w + x, h + y, confidences[i], self.labels[class_ids[i]]])
        return np.array(outputs_wrapper), 

    def Draw_detection(self, yolo_output, image):
        image_ = copy.deepcopy(image)
        labels_identified = []
        for bbox in yolo_output[0]:
            color = (255, 0, 0)
            bbox = list(np.array(bbox))
            x, y, x2, y2, object_id, Name = bbox
            labels_identified.append(Name)
            cv.rectangle(image_, (int(x), int(y)), (int(x2), int(y2)), color, 2)
            cv.putText(image_, Name, (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
        return image_, labels_identified
