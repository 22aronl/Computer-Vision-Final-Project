import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import time
from svm import SVM
from process_datasets import read_annotation_file, intersection_over_union
from sliding_window import image_pyramid, nme

base_path = 'B:CS376_Images/assignment5'
# base_path = '/Users/aaronlo/Downloads'
annotations_path = f"{base_path}/FDDB-folds/FDDB-fold-{{}}-ellipseList.txt"
images_path = f"{base_path}/originalPics/{{}}.jpg"

#calculates the confidence and correct values for the precision recall curve
def calculate_precision_recall(annotation_path, svm, iou_threshold=0.5):
    
    annotations = read_annotation_file(annotation_path)
    
    confidence = []
    correct = []
    total_faces = 0
    
    print(f'annotations {len(annotations)}')
    current_iteration = 0
    for annotation in annotations:
        print(f'current iteration {current_iteration} out of {len(annotations)}')
        current_iteration += 1
        image_path = annotation[0].strip()
        image = cv2.imread(images_path.format(image_path))
        grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #transforms the image to grey scale
        
        prediction = image_pyramid(grey_image, svm)    #gets the predictions from the image pyramid
        nme_predictions = nme(prediction)             #gets the non maxima suppression predictions

        faces = copy.deepcopy(annotation[1])
        for nms_pred in nme_predictions: #for each prediction
            max_iou = 0
            max_iou_face = None
            for face in faces: #find the face with the highest iou with existing faces
                iou = intersection_over_union(nms_pred[0:4], face)
                if(iou > max_iou):
                    max_iou = iou
                    max_iou_face = face
            
            if(max_iou > iou_threshold): #add to the list the confidence and correct values
                confidence.append(nms_pred[4])
                correct.append(1)
                faces.remove(max_iou_face)
            else:
                confidence.append(nms_pred[4])
                correct.append(0)
        total_faces += len(annotation[1])
        
    #save the confidence and correct values
    np.savez('precision_recall_values.npz', confidence=confidence, correct=correct, total_faces=total_faces)
    
    return confidence, correct, total_faces

#for cv2 face classifier
def calculate_precision_recall2(annotation_path, face_classifier, iou_threshold=0.5):
    
    annotations = read_annotation_file(annotation_path)
    
    confidence = []
    correct = []
    total_faces = 0
    
    print(f'annotations {len(annotations)}')
    current_iteration = 0
    for annotation in annotations:
        print(f'current iteration {current_iteration} out of {len(annotations)}')
        current_iteration += 1
        image_path = annotation[0].strip()
        image = cv2.imread(images_path.format(image_path))
        grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        face_c = face_classifier.detectMultiScale3(
        grey_image, scaleFactor=1.1, minNeighbors=5, minSize=(36, 20), outputRejectLevels=True
        )

        faces = copy.deepcopy(annotation[1])
        index = 0
        for nms_pred in face_c[0]:
            print(nms_pred)
            max_iou = 0
            max_iou_face = None
            for face in faces:
                iou = intersection_over_union(nms_pred[0:4], face)
                if(iou > max_iou):
                    max_iou = iou
                    max_iou_face = face
            
            if(max_iou > iou_threshold):
                confidence.append(face_c[2][index])
                correct.append(1)
                faces.remove(max_iou_face)
            else:
                confidence.append(face_c[2][index])
                correct.append(0)
            index += 1
        total_faces += len(annotation[1])
        
    #save the confidence and correct values
    
    return confidence, correct, total_faces

if __name__ == '__main__':

    svm = SVM()
    svm.load_model('C:/Users/AaronLo/Documents/cs376/Computer-Vision-Final-Project/weights_20231205_203814_epoch_30000.npz')
    # svm.load_model('/Users/aaronlo/Desktop/cs376/assignment5/weights/weights_20231204_050455_epoch_3000.npz')
    # face_classifier = cv2.CascadeClassifier(
    # cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    # )
    start_time = time.time()
    confidence, correct, total_faces = calculate_precision_recall(annotations_path.format(str(6).zfill(2)), svm)
    # confidence, correct, total_faces = calculate_precision_recall2(annotations_path.format(str(6).zfill(2)), face_classifier)
    end_time = time.time()
    print(f'Elapsed time {end_time - start_time}')
    npos = total_faces
    si = np.squeeze(np.argsort(-np.array(confidence), axis=0))
    
    nd = len(si)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    
    for d in range(nd):
        if(correct[si[d]] == 1):
            tp[d] = 1
        else:
            fp[d] = 1
    
    cum_fp = np.cumsum(fp)
    cum_tp = np.cumsum(tp)
    rec = cum_tp / npos
    prec = np.divide(cum_tp, (cum_fp + cum_tp)) #finds the points for precision and recall
    
    # Plot the precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(rec, prec, label='Precision-Recall Curve')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    plt.show()