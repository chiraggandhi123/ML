import cv2
import numpy as np
import matplotlib.pyplot as plt
nose_cascade=cv2.CascadeClassifier('/home/chirag/Desktop/chirag_ml/Assignment/Snapchat_filter/Traini/third-party/Nose18x15.xml')
cam=cv2.VideoCapture(0)
cnt=0
moustache=cv2.imread('/home/chirag/Desktop/chirag_ml/Assignment/Snapchat_filter/Traini/mustache.png')
#face_data=[]   
#u_name=input()
while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=nose_cascade.detectMultiScale(img,1.3,5)
    for (x,y,w,h) in faces:
        A=cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        #cv2.putText(gray,'chirag',(10,50),cv2.FONT_HERSHEY_COMPLEX,0,15,(255,255,255),2,cv2.LINE_AA)
        #face_section=img[(y:y+h),x]

        
        face_section=cv2.resize(face_section,(100,100))
        cv2.imshow('face',face_section)
        #A=A+moustache
        cv2.imshow('gray',img)
    k=cv2.waitKey(1)&0xFF
    if k==ord('q'):
        break
#print("TotalFace",len(face_data))
#face_data=np.array(face_data)
#face_data=face_data.reshape(face_data.shape[0],30000)
#print(face_data.shape)
#np.save('FaceData/'+u_name+'.npy',face_data)
#print('save at FaceData/'+u_name+'.npy')
#np.save()
cam.release()
#print(face_data)
cv2.destroyAllWindows()     
