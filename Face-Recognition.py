import cv2
import os
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
face_detector = cv2.CascadeClassifier("F:\Tarun\python\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0
while(True):
    ret, img = cam.read()
    #img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("C:\\Users\\hp\\Desktop\\pro\\dataset\\dataset\\USER." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break
        
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

#Train_data

import cv2
import numpy as np
from PIL import Image
import os
# Path for face image database
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("F:\Tarun\python\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
# function to get the images and label data
def getImagesAndLabels():
    imagePaths = [os.path.join('C:\\Users\\hp\\Desktop\\pro\\dataset\\dataset',f) for f in os.listdir('C:\\Users\\hp\\Desktop\\pro\\dataset\\dataset')]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels()
recognizer.train(faces, np.array(ids))
# Save the model into trainer/trainer.yml
recognizer.save('C:\\Users\\hp\\Desktop\pro\\trainer\\trainer.yml') #worked on Mac, but not on Pi
# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))


#FACE_RECOGNITION.PY
import cv2
import numpy as np
from PIL import Image
import os
# Path for face image database
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("F:\Tarun\python\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
# function to get the images and label data
def getImagesAndLabels():
    imagePaths = [os.path.join('C:\\Users\\hp\\Desktop\\pro\\dataset\\dataset',f) for f in os.listdir('C:\\Users\\hp\\Desktop\\pro\\dataset\\dataset')]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels()
recognizer.train(faces, np.array(ids))
# Save the model into trainer/trainer.yml
#recognizer.write('trainer/trainer.yml')
recognizer.save('C:\\Users\\hp\\Desktop\pro\\trainer\\trainer.yml') #worked on Mac, but not on Pi
# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

