import cv2
import numpy as np


#imread('image/face2.jpg')
cap = cv2.VideoCapture(0)
find_eyes = cv2.CascadeClassifier('eyes.xml')
face = cv2.CascadeClassifier('face.xml')

while True:
    success, img = cap.read()
    eye_frame = np.zeros(img.shape, np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, scaleFactor=2.5, minNeighbors=5)

    for(x, y, w, h) in faces:
        eyes = find_eyes.detectMultiScale(gray, scaleFactor=2.5, minNeighbors=3)
        if len(eyes) >= 2:
            ex1, ey1, ew1, eh1 = eyes[0]
            ex2, ey2, ew2, eh2 = eyes[1]

            if eyes[0][0] < eyes[1][0]:
                eyes = cv2.rectangle(img, (x, ey1), (x + w, ey1 + eh2), (0, 255, 0), thickness=2)
                cut_eye = eyes[ey1:ey1 + eh2, x:x + w]
                cut_eye = cv2.GaussianBlur(cut_eye, ksize=(49, 39), sigmaX=0)
                img[ey1:ey1 + eh2, x:x + w] = cut_eye
    
    cv2.imshow('Res', img)        

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.waitKey(0)