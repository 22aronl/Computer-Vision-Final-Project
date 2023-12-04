import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
from svm import SVM
from process_datasets import read_annotation_file, intersection_over_union
from sliding_window import image_pyramid, nme

base_path = 'B:CS376_Images/assignment5'
# base_path = '/Users/aaronlo/Downloads'
annotations_path = f"{base_path}/FDDB-folds/FDDB-fold-{{}}-ellipseList.txt"
images_path = f"{base_path}/originalPics/{{}}.jpg"

def calculate_precision_recall(annotation_path, svm, iou_threshold_step):
    
    annotations = read_annotation_file(annotation_path)
    
    true_positives = np.zeros(int(1 / iou_threshold_step) + 1)
    false_positives = np.zeros(int(1 / iou_threshold_step) + 1)
    false_negatives = np.zeros(int(1 / iou_threshold_step) + 1)
    
    print(f'annotations {len(annotations)}')
    annotations = annotations[0:3]
    current_iteration = 0
    for annotation in annotations:
        print(f'current iteration {current_iteration} out of {len(annotations)}')
        current_iteration += 1
        image_path = annotation[0].strip()
        image = cv2.imread(images_path.format(image_path))
        grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        prediction = image_pyramid(grey_image, svm)
        nme_predictions = nme(prediction)
        
        
        for i in range(0, (int(1 / iou_threshold_step) + 1)):
            faces = copy.deepcopy(annotation[1])
            iou_threshold = i * iou_threshold_step
            for nms_pred in nme_predictions:
                max_iou = 0
                max_iou_face = None
                for face in reversed(faces):
                    iou = intersection_over_union(nms_pred[0:4], face)
                    if(iou > max_iou):
                        max_iou = iou
                        max_iou_face = face
                
                if(max_iou > iou_threshold):
                    true_positives[i] += 1
                    faces.remove(max_iou_face)
                else:
                    false_positives[i] += 1
                    
            false_negatives[i] += len(faces)

    print(f'true_positives {true_positives}')
    print(f'false_positives {false_positives}')
    print(f'false_negatives {false_negatives}')
    
    precision = true_positives / (true_positives + false_positives) if np.any(true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if np.any(true_positives + false_negatives) > 0 else 0
    
    print(f'precision {precision}')
    print(f'recall {recall}')
    
    return precision, recall

if __name__ == '__main__':

    svm = SVM()
    svm.load_model('C:/Users/AaronLo/Documents/cs376/Computer-Vision-Final-Project/weights/weights_20231127_194326_epoch_25000.npz')


    precision_values, recall_values = calculate_precision_recall(annotations_path.format(str(6).zfill(2)), svm, 0.1)

    # Plot the precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall_values, precision_values, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)  
    
    plt.show()