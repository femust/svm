#!/usr/bin/env python3
    # TODO: This class should which Implements the following functionality:
    # - opencv HOGDescriptor combined with a sliding window
    # - perform detection at multiple scales, i.e. you need to scale the extracted patches when performing the detection
    # - non maximum suppression: eliminate detections using non-maximum-suppression based on the overlap area

import cv2
import math
import numpy as np
class CustomHogDetector:

    
    def __init__(self, trained_svm_name, custom = False):
        self.detection_width	= 64 # the crop width dimension
        self.detection_height = 128 # the crop height dimension
        self.window_stride = 32 # the stride size 
        self.scaleFactors = [1.2] # scale each patch down by this factor, feel free to try other values
        # You may play with different values for these two theshold values below as well 
        self.hit_threshold = 0 # detections above this threshold are counted as positive. 
        self.overlap_threshold = 0.3 # if the overlap between two detections is above this threshold, eliminate the one with the lower confidence score. 
        
        if (not custom):
            try:
                self.svm = cv2.ml.SVM_load(trained_svm_name)
            except:
                print("Missing files - SVM!")
                exit()
        if (custom):
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            # Some constants that you will be using in your implementation

    def is_inside(self, i, o):
        ix, iy, iw, ih = i
        ox, oy, ow, oh = o
        return ix > ox and ix + iw < ox + ow and iy > oy and iy + ih < oy + oh

    def is_bigger_than_overlap_threshold(self, r,q):
        rx, ry, rw, rh = r
        qx, qy, qw, qh = q

        area_r = rw * rh
        area_q = qw * qh

        area_intersect = (-max(rx,qx) + min(rx + rw , qx+qw))*(-max(ry,qy) + min(ry + rh, qy + qh))
        area_sum = area_r + area_q
        factor = area_sum / area_intersect

        if (factor > self.overlap_threshold):
            return True
        
        return False

    def detect(self, image, show = True):
        found_rects, found_weights = self.hog.detectMultiScale(image, winStride=(self.window_stride, self.window_stride), scale=self.scaleFactors[0], finalThreshold=self.hit_threshold)
        found_rects_filtered = []
        found_rects_nonfiltered = []
        found_weights_filtered = []
        for ri, r in enumerate(found_rects):
            for qi, q in enumerate(found_rects):
                found_rects_nonfiltered.append(r)
                if ri != qi and self.is_inside(r, q) and self.is_bigger_than_overlap_threshold(r,q):
                    break
            else:
                found_rects_filtered.append(r)
                found_weights_filtered.append(found_weights[ri])
        return found_rects_filtered, found_weights_filtered, found_rects_nonfiltered

    def resize_img(self,  img, width=-1, height=-1):
        h,w,_ = img.shape
        if height == -1:
            aspect_ratio = float(w) / h
            new_height = int(width / aspect_ratio)
            return cv2.resize(img, (width, new_height))
        elif width == -1:
            aspect_ratio = h / float(w)
            new_width = int(height / aspect_ratio)
            return cv2.resize(img, (new_width, height))

    def pyramid(self, img, scale=1.5, min_size=(30, 30)):
        yield img
        while True:
            w = int(img.shape[1] / scale)
            img = self.resize_img(img,w)
            if img.shape[0] < min_size[1] or img.shape[1] < min_size[0]:
                break
            yield img

    def sliding_window(self, img, window_size, step_size=8):
        h,w,_ = img.shape
        for y in range(0, h, step_size):
            for x in range(0, w, step_size):
                # yield the current window
                window = img[y:y + window_size[1], x:x + window_size[0]]
                if not (window.shape[0] != window_size[1] or window.shape[1] != window_size[0]):
                    yield (x, y, window)


    def non_max_suppression_fast(self, boxes, overlapThresh):
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
            idxs = np.delete(idxs, np.concatenate(([last],
                                                np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")

    def detectMultiScale(self, img):
        output_img = img.copy()

        current_scale = -1
        detections = []
        rescaling_factor = 1.25


        for resized in self.pyramid(img, scale=rescaling_factor):
            if current_scale == -1:
                current_scale = 1
            else:
                current_scale /= rescaling_factor

            rect_img = resized.copy()


            window_size = (self.detection_width, self.detection_height)
            step = math.floor(resized.shape[0] / 16)

            if step > 0:
                for (x, y, window) in self.sliding_window(resized, window_size, step_size=step):



                    hog = cv2.HOGDescriptor(); # default is 64 x 128
                    img_hog = cv2.resize(window, 
                                        (self.detection_width, self.detection_height), 
                                        interpolation = cv2.INTER_AREA)

                    hog_descriptor = hog.compute(img_hog)


                    if hog_descriptor is not None:

                        print("detecting with SVM ...")

                        retval, [result] = self.svm.predict(np.float32([hog_descriptor]))

                        print(result)
                        if result[0] == 1.0:
                            rect = np.float32([x, y, x + window_size[0], y + window_size[1]])
                            rect *= (1.0 / current_scale)
                            detections.append(rect)
        detections_nms = self.non_max_suppression_fast(np.int32(detections), 0.4)

        for rect in detections_nms:
            cv2.rectangle(output_img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)

        cv2.imshow('detected objects',output_img)
        key = cv2.waitKey(200) # wait 200ms





