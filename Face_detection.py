import cv2 as cv

img=cv.imread("photos/astronauts.jpg")
cv.imshow('Person',img)

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("Gray scale",gray)

#reading the haar cascade classifier
haar_cascade=cv.CascadeClassifier('haar_face.xml')

#face detection
faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=1)

print(f'Num of faces:',{len(faces_rect)})

for (x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),[0,255,0],2)

cv.imshow('Detected Faces',img)

#haar cascade for a video
# video=cv.VideoCapture(0)
# while True:
#     isOk,frame=video.read()
#     gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
#     video_face_detect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=1)
#     for(x,y,w,h) in video_face_detect:
#         cv.rectangle(frame,(x,y),(x+w,y+h),[0,255,0],2)

#     cv.imshow("Video face detect",frame)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

cv.waitKey(0)