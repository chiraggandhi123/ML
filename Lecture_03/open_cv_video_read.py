import cv2 as cv2
import numpy as np
face_cascade=cv2.CascadeClassifier('/home/chirag/Desktop/prateek bhaiya/machine-learning-june-2019/Lecture-03 FaceRecognition/haarcascade_frontalface_alt.xml')
cam=cv2.VideoCapture(0)
face_data=[]
cnt=0
user_name=input("Enter your name")
while True:
    ret, frame=cam.read()
    if ret==False:
        print("Something went wrong")
        continue
    keypressed=cv2.waitKey(1)&0xFF#waitkey returns 32 bit but char is 8 bit so bitmasking 32 bit in 8 bit
    if keypressed==ord('q'):#ord converts
        break
    #cv2.imshow('Video',frame[::,::,::])#[yaxis,xaxis,zaxis]
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray.reshape(160,640,3)
    #cv2.imshow('Video',gray)
    #cv2.imshow('RGB',frame)
    #horizontal=np.hstack(gray,frame)---gray[640,480] can't be stacked with rgb[640,480,3] because of size issues
    #BGR2GRAY it takes mean values accross all axis and places it on all dimensions
   # new_image=np.zeros((gray.shape))
    
   # new_image[:,:,0]=gray[:,:,0]
    #new_image[:,:,1]=gray[:,:,1]
    #new_image[:,:,2]=gray[:,:,2]
    #print(gray.shape)

#SNAPCHAT FILTERS
    #bright_image=frame*1+10#1 is alpha controls contrast and 10 is brightness that controls brightness
    #cv2.imshow('bright',bright_image)
#HARCASCADE-only determine front face and it crops the face part
    faces=face_cascade.detectMultiScale(frame,1.5,3)#1.5 is the rate at which rectangle grows
    print(faces)
    if(len(faces)==0):
        cv2.imshow('face',frame)
        continue
    for face in faces:
        x,y,w,h = face #(x,y)-top left (w,h)bottom right
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        
        #crop the face
        face_section=frame[y-10:y+h+10,x-10:x+w+10]
        face_section=cv2.resize(face_section,(100,100))
        if cnt%10==0:
            print('Taking frame',int(cnt/10))
            face_data.append(face_section)
        cnt+=1
        #print(face_data.shape)
        cv2.imshow("face",frame)
        cv2.imshow("face_section",face_section)
#save the face data and print
print("Total_face ",len(face_data))
face_data=np.array(face_data)
face_data=face_data.reshape((face_data.shape[0],30000))
np.save("FaceData/"+user_name+".npy",face_data)
print("Saved at FaceData/"+user_name+".npy")
print(face_data.shape)
c = np.fromfile("/home/chirag/Desktop/machine-learning-june-2019/FaceData/Chirag.npy",  dtype=np.int64)
print(c)
print(type(c))
cam.release()   
cv2.destroyAllWindows()