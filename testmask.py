import cv2
img=cv2.imread("masklalal.jpg")
cv2.imwrite("tmsk2.jpg",img)
img = cv2.GaussianBlur(img,(95,95),0)
ret,img = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
cv2.imwrite("tmsk1.jpg",img)
