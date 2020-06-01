#!/usr/bin/env python3
import cv2 
import numpy as np
import random
import os 

from custom_hog_detector import CustomHogDetector
# Global constants

# crop/patch dimensions for the training samples
width = 64
height = 128

num_negative_samples = 10 # number of negative samples per image
train_hog_path = '../train_hog_descs.dat' # the file to which you save the HOG descriptors of every patch
train_labels = '../labels_train.dat' # the file to which you save the labels of the training data
my_svm_filename = '../my_pretrained_svm.dat' # the file to which you save the trained svm 

#data paths
test_images_1 = '../data/task_1_testImages/'
path_train_2 = '../task_2_3_Data/01Train/'
path_test_2 = '../task_2_3_Data/02Test/'

#***********************************************************************************
# draw a bounding box in a given image
# Parameters:
# im: The image on which you want to draw the bounding boxes
# detections: the bounding box of the detections (people)
# returns None

def drawBoundingBox(image, detections, color = (0,255,0)):
    for ri, r in enumerate(detections):
        x, y, w, h = r
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        #text = '%.2f' % found_weights_filtered[ri]
        #cv2.putText(image, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def task1():
    print('Task 1 - OpenCV HOG')
    
    # Load images
    filelist = test_images_1 + "filenames.txt"
    filenames = []
    with open(filelist, "r") as f:
        filenames.append(f.read(-1).splitlines())

    custom_hog = CustomHogDetector()
    for fn in filenames[0]:
        print(os.path.join(test_images_1,fn), ' - ',)
        img_path = os.path.join(test_images_1,fn)
        image = cv2.imread(img_path)
        filtered, weights, found = custom_hog.detect(image, True)
        
        drawBoundingBox(image, filtered,(0,255,0))
        drawBoundingBox(image, found,(255,0,0))



    



def task2():

    print('Task 2 - Extract HOG features')

    random.seed()
    np.random.seed()

    # Load image names
  
    filelist_train_pos = path_train_2 + 'filenamesTrainPos.txt'
    filenames_train_pos = []
    with open(filelist_train_pos, "r") as f:
        filenames_train_pos.append(f.read(-1).splitlines())

    filelist_train_neg = path_train_2 + 'filenamesTrainNeg.txt'
    filenames_train_neg = []
    with open(filelist_train_neg, "r") as f:
        filenames_train_neg.append(f.read(-1).splitlines())

    hog = cv.HOGDescriptor()
    h_pos = []
    h_neg = []
    for pos_img,neg_img in zip(filenames_train_pos[0],filenames_train_neg[0]):
        im = cv.imread(path_train_2+"pos/"+pos_img)
        h,w,c = im.shape
        #Calc center crop
        ih2,iw2 = h//2,w//2
        h2,w2 = height//2, width//2
        y1,y2 = iw2-w2,iw2+w2
        x1,x2 = ih2-h2,ih2+h2
        crop_img = im[x1:x2,y1:y2]
        h = hog.compute(crop_img)
        h_pos.append(h)

        im = cv.imread(path_train_2+"neg/"+neg_img)
        h,w,c = im.shape
        h2,w2 = height//2, width//2
        for i in range(10):
            ih2,iw2 = np.random.randint(h2,h-h2),np.random.randint(w2,w-w2)
            y1,y2 = iw2-w2,iw2+w2
            x1,x2 = ih2-h2,ih2+h2
            crop_img = im[x1:x2,y1:y2]
            feat = hog.compute(crop_img)
            h_neg.append(feat)






def task3(): 
    print('Task 3 - Train SVM and predict confidence values')
      #TODO Create 3 SVMs with different C values, train them with the training data and save them
      # then use them to classify the test images and save the results
    

    filelist_testPos = path_test_2 + 'filenamesTestPos.txt'
    filelist_testNeg = path_test_2 + 'filenamesTestNeg.txt'
    




def task5():

    print ('Task 5 - Eliminating redundant Detections')
    

    # TODO: Write your own custom class myHogDetector 
    # Note: compared with the previous tasks, this task requires more coding
    
    my_detector = Custom_Hog_Detector(my_svm_filename)
   
    # TODO Apply your HOG detector on the same test images as used in task 1 and display the results

    print('Done!')
    cv.waitKey()
    cv.destroyAllWindows()






if __name__ == "__main__":

    # Task 1 - OpenCV HOG
    task1()

    # Task 2 - Extract HOG Features
    #task2()

    # Task 3 - Train SVM
    #task3()

    # Task 5 - Multiple Detections
    #task5()

