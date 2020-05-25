from keras.models import load_model
from MdlTrain import *
import numpy as np
import cv2

from model import autoencoder_B,autoencoder_A
from model import encoder, decoder_A, decoder_B

image_size = 64

encoder.load_weights("nwmdlt/encoder.h5")
decoder_A.load_weights("nwmdlt/decoder_A.h5")
decoder_B.load_weights("nwmdlt/decoder_B.h5")

imgsA=loadImgs(r"C:\Users\Q\Documents\Deepswap\Oout2")/255
imgsB=loadImgs(r"C:\Users\Q\Documents\Deepswap\TaOutT")/255


#imgsA=loadImgs(r"C:\Users\Q\Documents\Deepswap\Oout")/255.0
#imgsB=loadImgs(r"C:\Users\Q\Documents\Deepswap\TaOut")/255.0
a_faces = np.ndarray(shape=(5,128,128,3))
b_faces = np.ndarray(shape=(5,128,128,3))


##for i in range(10):
##    print(imgsA[i].shape)
##    print(a_faces[i].shape)
##    a_faces[i] = cv2.resize(imgsA[i], (64,64), interpolation = cv2.INTER_AREA)
##    #imgsA[i]=cv2.resize(imgsA[i],(64,64))
##    #a_faces[i]=imgsA[i]
##    b_faces[i] = cv2.resize(imgsB[i], (64,64), interpolation = cv2.INTER_AREA)
##    #imgsB[i]=cv2.resize(imgsB[i],(,64))
##    #b_faces[i]=imgsB[i]
    
wrap ,a_faces = get_training_data(imgsA,6)
wrap2 ,b_faces = get_training_data(imgsB,6)

#input()
# show original image
for (index, img) in enumerate(a_faces):
    winn="original_image1_"+str(index)
    cv2.namedWindow(winn)        
    cv2.moveWindow(winn, 10,index*170) 
    cv2.imshow(winn, img)

#a_faces = a_faces.astype('float32') / 255.
#wrap = wrap.astype('float32') / 255.

decoded_imgs = autoencoder_A.predict(a_faces)
#decoded_imgs = (decoded_imgs *255).astype(np.uint8)
print(decoded_imgs.shape)
for (index, img) in enumerate(decoded_imgs):
    winn="dec_image1_"+str(index)
    cv2.namedWindow(winn)        
    cv2.moveWindow(winn, 130,index*170) 
    cv2.imshow(winn, img)
decoded_imgs = autoencoder_B.predict(a_faces)
#decoded_imgs = (decoded_imgs*255).astype(np.uint8)

for (index, img) in enumerate(decoded_imgs):
    winn="dec_image4_"+str(index)
    cv2.namedWindow(winn)        
    cv2.moveWindow(winn, 250,index*170) 
    cv2.imshow(winn, img)


for (index, img) in enumerate(b_faces):
    winn="original_image2_"+str(index)
    cv2.namedWindow(winn)        
    cv2.moveWindow(winn, 370,index*170) 
    cv2.imshow(winn, img)

decoded_imgs = autoencoder_B.predict(b_faces)
#decoded_imgs = (decoded_imgs*255).astype(np.uint8)

for (index, img) in enumerate(decoded_imgs):
    winn="dec_image2_"+str(index)
    cv2.namedWindow(winn)        
    cv2.moveWindow(winn, 490,index*170)
    cv2.imshow(winn, img)
    #print(img)
    #cv2.imwrite("i"+str(index)+".jpg", img)


#a_faces = a_faces.astype('float32') / 255.
#wrap = wrap.astype('float32') / 255.

decoded_imgs = autoencoder_A.predict(b_faces)
#decoded_imgs = (decoded_imgs*255).astype(np.uint8)
for (index, img) in enumerate(decoded_imgs):
    winn="dec_image32_"+str(index)
    cv2.namedWindow(winn)        
    cv2.moveWindow(winn, 610,index*170) 
    cv2.imshow(winn, img)
    #cv2.imwrite("i"+str(index)+".jpg", img*255)
wrap ,a_faces = get_training_data(imgsA,6)
wrap2 ,b_faces = get_training_data(imgsB,6)

#input()
# show original image
for (index, img) in enumerate(a_faces):
    winn="original_image1312_"+str(index)
    cv2.namedWindow(winn)        
    cv2.moveWindow(winn, 730,index*170) 
    cv2.imshow(winn, img)

#a_faces = a_faces.astype('float32') / 255.
#wrap = wrap.astype('float32') / 255.

decoded_imgs = autoencoder_A.predict(a_faces)
#decoded_imgs = (decoded_imgs *255).astype(np.uint8)
print(decoded_imgs.shape)
for (index, img) in enumerate(decoded_imgs):
    winn="dec_imag32e1_"+str(index)
    cv2.namedWindow(winn)        
    cv2.moveWindow(winn, 850,index*170) 
    cv2.imshow(winn, img)
decoded_imgs = autoencoder_B.predict(a_faces)
#decoded_imgs = (decoded_imgs*255).astype(np.uint8)

for (index, img) in enumerate(decoded_imgs):
    winn="dec_imag324e4_"+str(index)
    cv2.namedWindow(winn)        
    cv2.moveWindow(winn, 970,index*170) 
    cv2.imshow(winn, img)


for (index, img) in enumerate(b_faces):
    winn="origi34nal_image2_"+str(index)
    cv2.namedWindow(winn)        
    cv2.moveWindow(winn, 1090,index*170) 
    cv2.imshow(winn, img)

decoded_imgs = autoencoder_B.predict(b_faces)
#decoded_imgs = (decoded_imgs*255).astype(np.uint8)

for (index, img) in enumerate(decoded_imgs):
    winn="dec_im123age2_"+str(index)
    cv2.namedWindow(winn)        
    cv2.moveWindow(winn, 1210,index*170)
    cv2.imshow(winn, img)
    #print(img)
    #cv2.imwrite("i"+str(index)+".jpg", img)


#a_faces = a_faces.astype('float32') / 255.
#wrap = wrap.astype('float32') / 255.

decoded_imgs = autoencoder_A.predict(b_faces)
#decoded_imgs = (decoded_imgs*255).astype(np.uint8)
for (index, img) in enumerate(decoded_imgs):
    winn="dec_im321age3_"+str(index)
    cv2.namedWindow(winn)        
    cv2.moveWindow(winn, 1330,index*170) 
    cv2.imshow(winn, img)
    cv2.imwrite("i"+str(index)+".jpg", img*255)



cv2.waitKey(0)
cv2.destroyAllWindows()
