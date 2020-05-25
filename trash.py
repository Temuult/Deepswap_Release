import cv2
from MdlTrain import *
batch_size = 1
imgsA=loadImgs(r"C:\Users\Q\Documents\Deepswap\temp")
warped_A, target_A = get_training_data( imgsA, batch_size )
cv2.imshow("1",warped_A[0])
print(warped_A.shape)
cv2.imshow("2",target_A[0])
print(target_A.shape)
