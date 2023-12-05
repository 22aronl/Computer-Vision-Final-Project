import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
from svm import SVM
from process_datasets import read_annotation_file, intersection_over_union
from sliding_window import image_pyramid, nme

# base_path = 'B:CS376_Images/assignment5'
base_path = '/Users/aaronlo/Downloads'
annotations_path = f"{base_path}/FDDB-folds/FDDB-fold-{{}}-ellipseList.txt"
images_path = f"{base_path}/originalPics/{{}}.jpg"

def calculate_precision_recall(annotation_path, svm, iou_threshold=0.5):
    
    annotations = read_annotation_file(annotation_path)
    
    confidence = []
    correct = []
    total_faces = 0
    
    print(f'annotations {len(annotations)}')
    annotations = annotations[0:15]
    current_iteration = 0
    for annotation in annotations:
        print(f'current iteration {current_iteration} out of {len(annotations)}')
        current_iteration += 1
        image_path = annotation[0].strip()
        image = cv2.imread(images_path.format(image_path))
        grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        prediction = image_pyramid(grey_image, svm)
        nme_predictions = nme(prediction)

        faces = copy.deepcopy(annotation[1])
        for nms_pred in nme_predictions:
            max_iou = 0
            max_iou_face = None
            for face in faces:
                iou = intersection_over_union(nms_pred[0:4], face)
                if(iou > max_iou):
                    max_iou = iou
                    max_iou_face = face
            
            if(max_iou > iou_threshold):
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

if __name__ == '__main__':

    svm = SVM()
    # svm.load_model('C:/Users/AaronLo/Documents/cs376/Computer-Vision-Final-Project/weights/weights_20231204_140212_epoch_60000.npz')
    svm.load_model('/Users/aaronlo/Desktop/cs376/assignment5/weights/weights_20231204_050455_epoch_3000.npz')

    confidence, correct, total_faces = calculate_precision_recall(annotations_path.format(str(6).zfill(2)), svm, 0.1)

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
    prec = np.divide(cum_tp, (cum_fp + cum_tp))
    
    # Plot the precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(rec, prec, label='Precision-Recall Curve')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)  
    
    plt.show()