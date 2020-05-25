from ApplyMdl import *

import numpy as np
import cv2

from model import autoencoder_B,autoencoder_A
from model import encoder, decoder_A, decoder_B

encoder.load_weights("nwmdlg/encoder.h5")
decoder_A.load_weights("nwmdlg/decoder_A.h5")
decoder_B.load_weights("nwmdlg/decoder_B.h5")

img=cv2.imread(r"C:\Users\Q\Documents\Deepswap\put.png")
res=applyswap(img,autoencoder_A)
cv2.imshow("1",res)
cv2.imwrite("2.jpg",res)

