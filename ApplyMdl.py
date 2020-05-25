from keras.models import load_model
from model import autoencoder_B,autoencoder_A
from model import encoder, decoder_A, decoder_B
from imutils import face_utils

#from MdlTrain import *
import numpy as np
import cv2
import dlib
import time
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

MOUTH_POINTS = list(range(48, 61))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
# Points used to line up the images.
ALIGN_POINTS = (RIGHT_EYE_POINTS + LEFT_EYE_POINTS + MOUTH_POINTS)


def get_landmark2(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    rects = detector(gray,0)
    for rect in rects:
        shape=predictor(gray, rect)
        mkrs = face_utils.shape_to_np(shape)
    return mkrs
def get_landmark(im):
    rects = detector(im, 1)
   
    if len(rects) == 0:
        raise NoFaces

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])
def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im
def applymodel(vidpath,outpath,mdlpath):
    try:
        encoder.load_weights(mdlpath+r"/encoder.h5")
        decoder_A.load_weights(mdlpath+r"/decoder_A.h5")
        decoder_B.load_weights(mdlpath+r"/decoder_B.h5")
        print("Model succesfully loaded")
    except:
        print("error no model found")

    cap = cv2.VideoCapture(vidpath)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    outfilenm=outpath+r"/out.avi"
    writer = cv2.VideoWriter(outfilenm,cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))
    fr=0

    for i in range(frames_count):
        stat,frame = cap.read()
        #try:
        #print(frame.shape)
        #input()
        #cv2.imshow("hasdasdj",frame)
        
        start_time = time.time()
        writer.write(applyswap(frame,autoencoder_A))
        print(str(fr)+"--- %s seconds ---" % (time.time() - start_time))
        fr=fr+1
    cap.release()
    writer.release()

def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur =im2_blur + 128 * (im2_blur <= 1.0)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                                                im2_blur.astype(np.float64))

def applyswap(bimg,autoencodernm):
    try:
        points=np.zeros((1,4,2),dtype=int)
        ranges=[6,10,26,17]
    
        lnm2=get_landmark(bimg)

        maskl = np.zeros(bimg.shape[:2], dtype=np.uint8)
    
        for index,i in enumerate(ranges):
            points[0,index]=(lnm2[i,0],lnm2[i,1])
        cv2.drawContours(maskl, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
        x,y,w,h = cv2.boundingRect(points)
        center=(int(x+w/2),int(y+h/2))
    
        nr=lnm2[8].item(1)-lnm2[29].item(1)
        nrb=nr*0.9

        sny=lnm2[29].item(0)-nr
        snx=lnm2[29].item(1)-int(nrb)
        eny=lnm2[29].item(0)+nr
        enx=lnm2[29].item(1)+int(2*nr-int(nrb))
    #=(int((sny+eny)/2),int((snx+enx)/2))
        fimg = bimg[snx:enx, sny:eny]/255
     
        fimgl = cv2.resize(fimg, (256,256), interpolation = cv2.INTER_AREA)
    
        faces = np.ndarray(shape=(1,256,256,3) , dtype=fimgl.dtype )
        faces[0]=fimgl
        fimgl = cv2.resize(fimg, (256,256), interpolation = cv2.INTER_AREA)
        fimgl=cv2.resize(fimgl,(128,128))
        faces = np.ndarray(shape=(1,128,128,3) , dtype=fimgl.dtype )
        faces[0]=fimgl
        #wrap ,face = get_training_data(faces,1)
     
        decoded_imgs = autoencodernm.predict(faces)
    
        dcdimg=decoded_imgs[0]*255
        normdcimg = np.zeros((dcdimg.shape[0:2]))
        normdcimg = cv2.normalize(dcdimg,  normdcimg, 0, 255, cv2.NORM_MINMAX)
        condcimg = cv2.convertScaleAbs(normdcimg)

        bfimg = cv2.convertScaleAbs(fimg*255)
        
        lndm1=get_landmark(condcimg)
        lndm=get_landmark(bfimg)
        
        mask = np.zeros(bfimg.shape[:2], dtype=np.uint8)
        kernel = np.ones((7,7), np.uint8)
        for index,i in enumerate(ranges):
            points[0,index]=(lndm[i,0],lndm[i,1])
        cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)

        height, width = bfimg.shape[:2]
        #@larger=cv2.resize(condcimg,(height,width))

        M=transformation_from_points(lndm[ALIGN_POINTS],lndm1[ALIGN_POINTS])

        larger = warp_im(condcimg, M, bfimg.shape)
        mask = cv2.dilate(mask, kernel, iterations=1)
        maskl = cv2.dilate(maskl, kernel, iterations=1)
        #cv2.imwrite("l.jpg",larger)



        #MASK
        kernel = np.ones((15,15), np.uint8)
        #mask = cv2.erode(mask, kernel, iterations=1)
        #mask = cv2.GaussianBlur(mask,(21,21),0)
        mask = cv2.GaussianBlur(mask,(105,105),0)
        ret,mask = cv2.threshold(mask,125,255,cv2.THRESH_BINARY)
        kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])


        #Bitwise
        larger = cv2.GaussianBlur(larger, (3, 3), 0)
        
        blur_img = cv2.GaussianBlur(larger, (0, 0), 5)
        usm = cv2.addWeighted(larger, 1.5, blur_img, -0.5, 0)

        #h, w = larger.shape[:2]
        #result = np.zeros([h, w*2, 3], dtype=larger.dtype)
        #result[0:h,0:w,:] = larger
        #result[0:h,w:2*w,:] = usm
        #print(larger.shape)
        #print(result.shape)
        
        cv2.imwrite("fgd.jpg",usm)

        #larger = usm
        #larger = cv2.filter2D(larger, -1, kernel_sharpening)
        
        res = cv2.bitwise_and(larger,larger,mask = mask)
        
    
        #height, width = fimg.shape[:2]
        #larger=cv2.resize(res,(height,width))
    
        #lmask=cv2.resize(mask,(height,width))
        #print(center)
        
        ret,alpha = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
    
        b_channel, g_channel, r_channel = cv2.split(res)
        res = cv2.merge((b_channel, g_channel, r_channel, alpha))
        alpha_s = res[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
    
        bimgcp=bimg.copy()
    
        y1=snx
        y2=enx
        x1=sny
        x2=eny
        for c in range(0, 3):
            bimg[y1:y2, x1:x2, c] = (alpha_s * res[:, :, c] +
                              alpha_l * bimg[y1:y2, x1:x2, c])
        #maskl= np.zeros(bimg.shape[:2], dtype=np.uint8)
        #maskl[y1:y1+mask.shape[0], x1:x1+mask.shape[1]] = mask
        #cv2.imwrite("bing3.jpg",bimgcp)
        #maskl = cv2.erode(maskl, kernel, iterations=1)
        #maskl = cv2.GaussianBlur(maskl,(21,21),0)
        maskl = cv2.GaussianBlur(maskl,(105,105),0)
        ret,maskl = cv2.threshold(maskl,135,255,cv2.THRESH_BINARY)
        #cv2.imwrite("masklarge.jpg",maskl)
        #cv2.imwrite("res.jpg",bimg)
        output = cv2.seamlessClone(bimg, bimgcp, maskl, center, cv2.NORMAL_CLONE)
       
        return output
    except Exception as e:
        print(e)
        return bimg
