#!/usr/bin/env python3
    # TODO: This class should which Implements the following functionality:
    # - opencv HOGDescriptor combined with a sliding window
    # - perform detection at multiple scales, i.e. you need to scale the extracted patches when performing the detection
    # - non maximum suppression: eliminate detections using non-maximum-suppression based on the overlap area

import cv2

class CustomHogDetector:

    # Some constants that you will be using in your implementation
    detection_width	= 64 # the crop width dimension
    detection_height = 128 # the crop height dimension
    window_stride = 32 # the stride size 
    scaleFactors = 1.2 # scale each patch down by this factor, feel free to try other values
    # You may play with different values for these two theshold values below as well 
    hit_threshold = 0 # detections above this threshold are counted as positive. 
    overlap_threshold = 0.3 # if the overlap between two detections is above this threshold, eliminate the one with the lower confidence score. 
    
    def __init__(self):#, trained_svm_name):
        #load the trained SVM from file trained_svm_name
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
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

    def detect(self, image_path, show = True):
        image = cv2.imread(image_path)
        found_rects, found_weights = self.hog.detectMultiScale(image, winStride=(self.window_stride, self.window_stride), scale=self.scaleFactors, finalThreshold=self.hit_threshold)
        found_rects_filtered = []
        found_weights_filtered = []
        for ri, r in enumerate(found_rects):
            for qi, q in enumerate(found_rects):
                if ri != qi and self.is_inside(r, q) and self.is_bigger_than_overlap_threshold(r,q):
                    break
            else:
                found_rects_filtered.append(r)
                found_weights_filtered.append(found_weights[ri])
        if show:
            for ri, r in enumerate(found_rects_filtered):
                x, y, w, h = r
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
                text = '%.2f' % found_weights_filtered[ri]
                cv2.putText(image, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
            cv2.imshow('People', image)
            cv2.waitKey(0)








