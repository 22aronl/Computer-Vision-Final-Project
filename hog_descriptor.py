import cv2
import numpy as np

#needs gray scaled input patch
def calculate_gradient(patch):
    
    dx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=1)
    dy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=1)

    magnitude = np.sqrt(dx*dx + dy*dy)
    direction = np.arctan2(dy, dx)

    return magnitude, direction

def calculate_histogram(magnitude, direction, bins=9):
    
    histogram = np.zeros(bins)
    angle = np.pi / bins

    for i in range(bins):
        angles = (i * angle <= direction) & (direction < ((i + 1) * angle))
        histogram[i] = np.sum(magnitude[angles])

    return histogram

#l2 normalization
def normalize_histogram(histogram):
    norm = np.linalg.norm(histogram)
    if norm == 0:
        norm = 1e-5
    return histogram / norm

# the patch should be scaled to 96 / 64 (tbd), but it should be divisible by 16 x 16, it should also be greyscaled
def extract_hog_descriptor(patch, cell_size=(8, 8), block_size=(2, 2), bins=9):
    
    histogram_points = []
    feature_vector = []
    
    assert(patch.shape[0] % (cell_size[0] * block_size[0]) == 0)
    assert(patch.shape[1] % (cell_size[1] * block_size[1]) == 0)
    assert(len(patch.shape) == 2) # gray scale image
    # cv2.imshow("img", (patch))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    for i in range(0, patch.shape[0], cell_size[0]):
        temp = []
        for j in range(0, patch.shape[1], cell_size[1]):
            cell_magnitude, cell_direction = calculate_gradient(patch[i:i+cell_size[0], j:j+cell_size[1]])
            cell_histogram = calculate_histogram(cell_magnitude, cell_direction, bins)
            temp.append(cell_histogram)
        histogram_points.append(temp)
        
    histogram_points = np.array(histogram_points)
    

    for i in range(0, histogram_points.shape[0] - block_size[0] + 1, block_size[0]):
        for j in range(0, histogram_points.shape[1] - block_size[1] + 1, block_size[1]):
            block = histogram_points[i:i+block_size[0], j:j+block_size[1], :].flatten()
            feature_vector.append(normalize_histogram(block))
            
    feature_vector = np.array(feature_vector).flatten()
    
    return feature_vector

