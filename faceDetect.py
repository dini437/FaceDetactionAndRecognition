import numpy as np
import cv2
import pickle


face_casecade = cv2.CascadeClassifier('C:/Users/DiniDev/PycharmProjects/FaceDetectReg/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("C:/Users/DiniDev/PycharmProjects/FaceDetectReg/trainner.yml")

labels = {"person_name": 1}
with open("C:/Users/DiniDev/PycharmProjects/FaceDetectReg/labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items() }


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_casecade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x, y, w, h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] #region of interest cordinate 1 (y start , y end)
        roi_color = frame[y:y+h, x:x+w] #Cordinate 2 (x start, x end)

        id_, conf = recognizer.predict(roi_gray)
        if conf>=45 : #and conf <=85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame,name, (x,y), font, 1, color, stroke, cv2.LINE_AA)


        img_item = "my_img.png"
        cv2.imwrite(img_item, roi_gray) #imgwrite

        color = (255,0,0)
        stroke = 2
        width = x + w
        height = y + h
        cv2.rectangle(frame, (x,y), (width, height), color, stroke)
    cv2.imshow('frame', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows