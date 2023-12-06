import cv2
import matplotlib.pyplot as plt



# the face detection code for a haar cascade classifier for faces from cv2
if __name__ == '__main__':
    image = cv2.imread("B:CS376_Images/assignment5/originalPics/2003/01/21/big/img_1071.jpg")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    face = face_classifier.detectMultiScale3(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40), outputRejectLevels=True
    )
    i = 0
    for (x, y, w, h) in face[0]:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
        print(face[2][i])
        
    cv2.imshow("Image with Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()