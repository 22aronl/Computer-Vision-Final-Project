import cv2

from process_datasets import resize_image, intersection_over_union_area
from hog_descriptor import extract_hog_descriptor
from svm import SVM

#greyscaled image
def sliding_window(image, window_size, classifier: SVM, threshold=0.0006, step_size=(8, 8)):
    height, width = image.shape
    predictions = []
    
    for i in range(0, height - window_size[1] + 1, step_size[1]):
        for j in range(0, width - window_size[0] + 1, step_size[0]):
            # print(f'{i} {j} {height} {width} {window_size[1]} {window_size[0]}')
            window = image[i:i + window_size[1], j:j + window_size[0]]
            # print(f'{len(window)} and {len(window[0])}')
            resized_window = resize_image(window)
            
            window_hog = extract_hog_descriptor(resized_window)
            prediction = classifier.predict(window_hog)
            
            if prediction > threshold:
                predictions.append([j, i, window_size[0], window_size[1], prediction])
                
    return predictions

def image_pyramid(image, classifier: SVM, aspect_ratio=1.5, min_window_size=(36, 24)):
    original_height, original_width = image.shape
    total_predictions = []
    
    min_height, min_width = min_window_size
    scale = 1.0

    while original_height >= min_height and original_width >= min_width:
        window_size = (min_width, min_height)
        print(f'current Window size {window_size} min_width {min_width} min_height = {min_height}')
        # breakpoint()
        window_predictions = sliding_window(image, window_size, classifier)
        total_predictions.extend(window_predictions)

        min_width = int(min_width * aspect_ratio)
        min_height = int(min_width * aspect_ratio)

    return total_predictions

def check_area(box1, box2, area_inter, area_threshold):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    area1 = w1 * h1
    area2 = w2 * h2
    
    return abs(1 - area_inter / area1) < area_threshold or abs(1 - area_inter / area2) < area_threshold

def add_box(boxes, box, iou_threshold=0.4, area_threshold=0.3):
    for b in boxes:
        iou, area_inter = intersection_over_union_area(b[0:4], box[0:4])
        if(iou > iou_threshold or check_area(b[0:4], box[0:4], area_inter, area_threshold)):
            return False
    return True

def nme(boxes, iou_threshold=0.4):
    new_boxes = []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    
    for box in boxes:
        if(add_box(new_boxes, box, iou_threshold)):
            new_boxes.append(box)
            
    return new_boxes
    

def overlay_boxes(image, boxes, color=(0, 0, 255), thickness=2):

    image_with_boxes = image.copy()

    for box in boxes:
        x, y, w, h, pred  = box
        print(f'pred value {pred}')
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), color, thickness)

    return image_with_boxes

if __name__ == '__main__':  
    svm = SVM()
    svm.load_model('/Users/aaronlo/Desktop/cs376/assignment5/weights/weights_20231126_213700_epoch_150000.npz')
    
    img = cv2.imread("/Users/aaronlo/Downloads/originalPics/2003/05/01/big/img_664.jpg")
    grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    total_predictions = image_pyramid(grey_img, svm)
    
    print(f'Number of predictions{len(total_predictions)}')
    nme_predictions = nme(total_predictions)
    image_with_boxes = overlay_boxes(img, nme_predictions)

    # Display the image with bounding boxes
    cv2.imshow("Image with Bounding Boxes", image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
            