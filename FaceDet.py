from imutils import face_utils
import cv2
import sys
import os
import dlib
import numpy
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def get_landmarks(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    rects = detector(gray,0)
    for rect in rects:
        shape=predictor(gray, rect)
        mkrs = face_utils.shape_to_np(shape)
    return mkrs
def extract(path,picsP,evrfr):
    makedir(picsP)
    try:
        cap=cv2.VideoCapture(path)
        print('Found input video')
    except:
        print('Unable to find input video')
    l=-1
    while(cap.isOpened()):
        l+=1
        ret, frame = cap.read()
        if ret==False:
            break
        if(l%int(evrfr)==0):
            cv2.imwrite(picsP+'/img'+str(l)+'.jpg',frame)
        
    cap.release()
    
def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = numpy.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords
def get_landmarks312(im):
    #gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    rects = detector(gray,0)
    for rect in rects:
        shape=predictor(gray, rect)
        mkrs = face_utils.shape_to_np(shape)
        #mkrs=shape_to_np(shape)
        #print(mkrs)
        #lnmrks=numpy.zeros((68,2),dtype=int)
        #for i, (x, y) in enumerate(shape):
            #cv2.circle(im, (x, y), 1, (0, 0, 255), -1)
            #cv2.imread(str(i),im)
            #input()
            #lnmrks[i]=([x,y])
    return mkrs
def getfiles(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpg' in file:
                files.append(os.path.join(r, file))
    return files
def generate(Pth):
    j=0
    facesP=Pth+"/temp"
    files=getfiles(facesP)
    for i,name in enumerate(files):
        img = cv2.imread(name)
        try:
            lndm=get_landmarks(img)
            nr=lndm[8].item(1)-lndm[29].item(1)
            nrb=nr*0.9
            sny=lndm[29].item(0)-nr
            snx=lndm[29].item(1)-int(nrb)
            eny=lndm[29].item(0)+nr
            enx=lndm[29].item(1)+int(2*nr-int(nrb))
            #nr
            #color = (255, 0, 0) 
            #img = cv2.rectangle(img, (sny,snx), (eny,enx), color, 2)
        
            faceimg = img[snx:enx, sny:eny]
            faceimg = cv2.resize(faceimg, (256,256), interpolation = cv2.INTER_AREA)
            j+=1
            cv2.imwrite(Pth+'/face'+str(j)+'.jpg', faceimg)
        except:
            print("No face found")
            continue
def clean(P):
    tempfls=getfiles(P)
    for fnm in tempfls:
        os.remove(fnm)
 
def makedir(name):
    try:
        os.stat(name)
    except:
        os.mkdir(name)
