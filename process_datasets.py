import os
import numpy as np
import cv2

from hog_descriptor import extract_hog_descriptor

def convert_to_rectangular(major_axis_radius, minor_axis_radius, angle, center_x, center_y):
    ux = major_axis_radius * np.cos(angle)
    uy = major_axis_radius * np.sin(angle)
    vx = minor_axis_radius * np.cos(angle + np.pi/2)
    vy = minor_axis_radius * np.sin(angle + np.pi/2)
    
    half_width = np.sqrt(ux*ux + vx*vx)
    half_height = np.sqrt(uy*uy + vy*vy)
    
    if(center_x - half_width < 0):
        half_width = center_x
    if(center_y - half_height < 0):
        half_height = center_y
        
    
    return [center_x-half_width, center_y-half_height, 2*half_width, 2*half_height]

def intersection_over_union_area(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    
    x_inter = max(x1, x2)
    y_inter = max(y1, y2)
    w_inter = min(x1 + w1, x2 + w2) - x_inter
    h_inter = min(y1 + h1, y2 + h2) - y_inter
    
    area_inter = max(0, w_inter) * max(0, h_inter)
    
    area_union = w1 * h1 + w2 * h2 - area_inter

    iou = area_inter / area_union if area_union > 0 else 0

    return iou, area_inter

def intersection_over_union(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    
    x_inter = max(x1, x2)
    y_inter = max(y1, y2)
    w_inter = min(x1 + w1, x2 + w2) - x_inter
    h_inter = min(y1 + h1, y2 + h2) - y_inter
    
    area_inter = max(0, w_inter) * max(0, h_inter)
    
    area_union = w1 * h1 + w2 * h2 - area_inter

    iou = area_inter / area_union if area_union > 0 else 0

    return iou

def generate_random_patch(image_shape, aspect_ratio=1.5, min_size = 20):
    image_height, image_width = image_shape
    img_aspect_ratio = image_height / image_width
    
    if img_aspect_ratio > aspect_ratio:
        patch_width = np.random.randint(min_size, image_width)
        patch_height = int(patch_width * aspect_ratio)
    else:
        patch_height = np.random.randint(min_size, image_height)
        patch_width = int(patch_height / aspect_ratio)
    # breakpoint()
    if(image_width - patch_width <= 0):
        print(f'{image_height} {image_width} {patch_height} {patch_width}')
    x = np.random.randint(0, image_width - patch_width)
    y = np.random.randint(0, image_height - patch_height)
    
    return [x, y, patch_width, patch_height]

def generate_false_patches(num_patches, image, true_patches, image_size=(96, 64), iou_threshold=0.3, max_attempts_scale=5):
    aspect_ratio = image_size[0] / image_size[1]
    false_patches = []
    successful_patches = 0
    for _ in range(num_patches * max_attempts_scale):
        patch = generate_random_patch([len(image), len(image[0])], aspect_ratio)
        
        # cv2.imshow("img", image[patch[1]:patch[1]+patch[3], patch[0]:patch[0]+patch[2]])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        iou = 0
        for true_patch in true_patches:
            iou = max(iou, intersection_over_union(true_patch, patch))
            
        if iou < iou_threshold:
            patch_hog = extract_hog_descriptor(resize_image(image[patch[1]:patch[1]+patch[3], patch[0]:patch[0]+patch[2]]))
            
            false_patches.append(patch_hog)
            successful_patches += 1
            if successful_patches == num_patches:
                break
        
    return false_patches

def extract_intended_ratio(center_x, center_y, half_width, half_height, target_ratio=1.5):
    aspect_ratio = half_height / half_width
    # print(f'old {half_height} {half_width}')
    if aspect_ratio > target_ratio:
        half_height = int(half_width * target_ratio)
    else:
        half_width = int(half_height / target_ratio)
    # print(f'{aspect_ratio} {half_height} {half_width} and new {half_height / half_width}')
    
    return center_x, center_y, half_width, half_height
    

def read_annotation_file(file_path):
    annotations = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            image_path = lines[i].strip()
            num_faces = int(lines[i+1].strip())
            faces_list = []
            for faces in range(num_faces):
                face_info = lines[i+2+faces].strip().split(' ')
                major_axis_radius = float(face_info[0])
                minor_axis_radius = float(face_info[1])
                angle = float(face_info[2])
                center_x = float(face_info[3])
                center_y = float(face_info[4])
                faces_list.append(convert_to_rectangular(major_axis_radius, minor_axis_radius, angle, center_x, center_y))
            annotations.append([image_path, faces_list])
            i += 2 + num_faces
            
    return annotations

# base_path = 'B:CS376_Images/assignment5'
base_path = '/Users/aaronlo/Downloads'
annotations_path = f"{base_path}/FDDB-folds/FDDB-fold-{{}}-ellipseList.txt"
images_path = f"{base_path}/originalPics/{{}}.jpg"

def read_images_with_annotations(annotation_path, target_ratio=1.5, false_scaling=9, testing=False):
    
    true_patches = []
    false_patches = []
    
    annotations = read_annotation_file(annotation_path)
    
    for annotation in annotations:
        image_path = annotation[0].strip()
        image = cv2.imread(images_path.format(image_path))
        grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        for face in annotation[1]:
            left_x = int(face[0])
            left_y = int(face[1])
            width = int(face[2])
            height = int(face[3])
        
            
            if(left_x+width > grey_image.shape[1]):
                width = grey_image.shape[1] - left_x
            if(left_y+height > grey_image.shape[0]):
                height = grey_image.shape[0] - left_y
                
            if(height < 10):
                continue
                
            left_x, left_y, width, height = extract_intended_ratio(left_x, left_y, width, height, target_ratio=target_ratio)
            true_patch = grey_image[left_y:left_y+height, left_x:left_x+width]
            # print(left_x, left_y, width, height, grey_image.shape, true_patch.shape, left_x-width, left_x+width)
            true_patch = resize_image(true_patch)
            true_patch_histogram = extract_hog_descriptor(true_patch)
        
            true_patches.append(true_patch_histogram)
            if not testing:
                small_shift = 4
                if(left_y - small_shift >= 0):
                    true_patch = grey_image[left_y-small_shift:left_y+height-small_shift, left_x:left_x+width]
                    true_patch = resize_image(true_patch)
                    true_patch_histogram = extract_hog_descriptor(true_patch)
                    true_patches.append(true_patch_histogram)
                    
                if(left_y + small_shift + height < grey_image.shape[0]):
                    true_patch = grey_image[left_y+small_shift:left_y+height+small_shift, left_x:left_x+width]
                    true_patch = resize_image(true_patch)
                    true_patch_histogram = extract_hog_descriptor(true_patch)
                    true_patches.append(true_patch_histogram)
                    
                if(left_x - small_shift >= 0):
                    true_patch = grey_image[left_y:left_y+height, left_x-small_shift:left_x+width-small_shift]
                    true_patch = resize_image(true_patch)
                    true_patch_histogram = extract_hog_descriptor(true_patch)
                    true_patches.append(true_patch_histogram)
                    
                if(left_x + small_shift + width < grey_image.shape[1]):
                    true_patch = grey_image[left_y:left_y+height, left_x+small_shift:left_x+width+small_shift]
                    true_patch = resize_image(true_patch)
                    true_patch_histogram = extract_hog_descriptor(true_patch)
                    true_patches.append(true_patch_histogram)
                
            
        false_patches.extend(generate_false_patches(20-len(annotation[1]), grey_image, annotation[1]))
    
    return true_patches, false_patches

# list of numbers of the annotations to be read
def read_image_set(image_set_path, true_weighting=1, testing=False):
    true_patches = []
    false_patches = []
    
    for path in image_set_path:
        sub_true_patches, sub_false_patches = read_images_with_annotations(annotations_path.format(str(path).zfill(2)), testing=testing)
        true_patches.extend(sub_true_patches)
        false_patches.extend(sub_false_patches)
    
    data_set = true_patches + false_patches
    
    y_true = [1 * true_weighting] * len(true_patches)
    y_true.extend([-1] * len(false_patches))
    
    print(f'dataset {len(true_patches)}, {len(false_patches)}, {len(data_set)}')
    # breakpoint()
    
    return np.array(data_set), np.array(y_true)
    
def read_train_set():
    return read_image_set(range(1, 8), true_weighting=1)

def read_test_set():
    return read_image_set(range(9, 10), testing=True)

def resize_image(image, target_size=(64, 96)):
    aspect_ratio = image.shape[0] / image.shape[1]
    if(abs(aspect_ratio - target_size[1] / target_size[0]) > 0.2):
        print(f'{aspect_ratio} {target_size} {image.shape}')
        # breakpoint()
        
    assert(abs(aspect_ratio - target_size[1] / target_size[0]) < 0.2)
    
    # if aspect_ratio > 1:
    #     new_width = target_size
    #     new_height = int(target_size / aspect_ratio)
    # else:
    #     new_width = int(target_size * aspect_ratio)
    #     new_height = target_size

    # Choose the appropriate interpolation method based on whether it's upscaling or downscaling
    if target_size[1] > image.shape[0] or target_size[0] > image.shape[1]:
        interpolation = cv2.INTER_LINEAR  # or cv2.INTER_CUBIC for higher quality
    else:
        interpolation = cv2.INTER_AREA

    resized_image = cv2.resize(image, target_size, interpolation=interpolation)

    return resized_image
    
    
if __name__ == '__main__':
    read_image_set([1])