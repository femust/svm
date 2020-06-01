#!/usr/bin/env python3
import cv2 
import cv2 as cv
import numpy as np
import random
import os 
import matplotlib.pyplot as plt

from custom_hog_detector import CustomHogDetector
# Global constants

# crop/patch dimensions for the training samples
width = 64
height = 128

dir_path = '/home/andrzej/studies/cv2/homework3/data/'

num_negative_samples = 10 # number of negative samples per image
# train_hog_path = dir_path'/train_hog_descs.dat' # the file to which you save the HOG descriptors of every patch
# train_labels = '../labels_train.dat' # the file to which you save the labels of the training data
# my_svm_filename = '../my_pretrained_svm.dat' # the file to which you save the trained svm 

#data paths
test_images_1 = dir_path + 'task_1_testImages/'
path_train_2 = dir_path + 'task_2_3_data/train/'
path_test_2 = dir_path  + 'task_2_3_data/test/'

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
    filelist = test_images_1 + 'filenames.txt'
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

    for pos_img in filenames_train_pos[0]:
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
    h_neg = []  
    for neg_img in filenames_train_neg[0]:
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
    hog_feats = np.concatenate([np.array(h_pos),np.array(h_neg)])
    print(hog_feats.shape, "saved features")
    np.save("data/train_hog_descs.npy",hog_feats)    



def calculate_precision_recall(prediction, labels):
    prediction_true = prediction == labels
    prediction_false = prediction != labels

    TP = np.count_nonzero(prediction_true)
    FP = np.count_nonzero(prediction_false[labels == 1])
    FN = np.count_nonzero(prediction_false[labels == 0])
    precission = TP / (TP + FP)
    recall = TP / (FN + TP)
    return precission, recall

def task3(): 
    from sklearn.svm import SVC 
    print('Task 3 - Train SVM and predict confidence values')
      #TODO Create 3 SVMs with different C values, train them with the training data and save them
      # then use them to classify the test images and save the results
    

    filelist_test_pos = path_test_2 + 'filenamesTestPos.txt'
    filenames_test_pos = []
    with open(filelist_test_pos, "r") as f:
        filenames_test_pos.append(f.read(-1).splitlines())

    filelist_test_neg = path_test_2 + 'filenamesTestNeg.txt'
    filenames_test_neg = []
    with open(filelist_test_neg, "r") as f:
        filenames_test_neg.append(f.read(-1).splitlines())

    hog_features = np.squeeze(np.load("data/train_hog_descs.npy"))
    labels = np.concatenate((np.ones(500),np.zeros(4000)))
    print(hog_features.shape,labels.shape)
    # Shuffle Samples
    rand = np.random.RandomState(321)
    shuffle = rand.permutation(hog_features.shape[0])
    hog_features = hog_features[shuffle]
    labels = labels[shuffle]    
    
    clf001  = SVC(C=0.01, probability=True)
    clf1    = SVC(C=1, probability=True)  
    clf100  = SVC(C=100, probability=True)      
    clf001.fit(hog_features,labels)
    clf1.fit(hog_features,labels)
    clf100.fit(hog_features,labels)

    hog = cv.HOGDescriptor()
    h_pos = []

    for pos_img in filenames_test_pos[0]:
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
    h_neg = []  
    for neg_img in filenames_test_neg[0]:
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
    hog_feats = np.squeeze(np.concatenate([np.array(h_pos),np.array(h_neg)]))
    labels_test = np.concatenate((np.ones(len(h_pos)),np.zeros(len(h_neg))))

    y001 = clf001.predict(hog_feats)
    precision001, recall001 = calculate_precision_recall(y001, labels_test)

    y1   = clf1.predict(hog_feats)
    precision1, recall1 = calculate_precision_recall(y1, labels_test)

    y100 = clf100.predict(hog_feats)
    precision100, recall100 = calculate_precision_recall(y100, labels_test)

    precision_x = [precision001, precision1, precision100]
    recall_y = [recall001, recall1, recall100]
    plt.plot(precision_x,recall_y)
    plt.show()








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
    #task1()

    # Task 2 - Extract HOG Features
    task2()

    # Task 3 - Train SVM
    task3()

    # Task 5 - Multiple Detections
    #task5()

