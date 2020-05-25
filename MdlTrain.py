import os
import sys
import keras

import cv2
import numpy
from FaceDet import getfiles
from umeyama import umeyama
from model import autoencoder_A
from model import autoencoder_B
from model import encoder, decoder_A, decoder_B
random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.01,
    'shift_range': 0.05,
    'random_flip': 0.4,
    }
def random_transform( image, rotation_range, zoom_range, shift_range, random_flip ):
    h,w = image.shape[0:2]
    rotation = numpy.random.uniform( -rotation_range, rotation_range )
    scale = numpy.random.uniform( 1 - zoom_range, 1 + zoom_range )
    tx = numpy.random.uniform( -shift_range, shift_range ) * w
    ty = numpy.random.uniform( -shift_range, shift_range ) * h
    mat = cv2.getRotationMatrix2D( (w//2,h//2), rotation, scale )
    mat[:,2] += (tx,ty)
    result = cv2.warpAffine( image, mat, (w,h), borderMode=cv2.BORDER_REPLICATE )
    if numpy.random.random() < random_flip:
        result = result[:,::-1]
    return result

# get pair of random warped images from aligened face image
def random_warp( image ):
    #cv2.imshow("321",image)
    assert image.shape == (256,256,3)
    range_ = numpy.linspace( 128-120, 128+120, 5 )
    mapx = numpy.broadcast_to( range_, (5,5) )
    mapy = mapx.T

    mapx = mapx + numpy.random.normal( size=(5,5), scale=5 )
    mapy = mapy + numpy.random.normal( size=(5,5), scale=5 )

    interp_mapx = cv2.resize( mapx, (160,160) )[16:144,16:144].astype('float32')
    interp_mapy = cv2.resize( mapy, (160,160) )[16:144,16:144].astype('float32')

    warped_image = cv2.remap( image, interp_mapx, interp_mapy, cv2.INTER_LINEAR )
    #print(warped_image.shape)
    src_points = numpy.stack( [ mapx.ravel(), mapy.ravel() ], axis=-1 )
    dst_points = numpy.mgrid[0:130:32,0:130:32].T.reshape(-1,2)
    mat = umeyama( src_points, dst_points, True )[0:2]

    target_image = cv2.warpAffine( image, mat, (128,128) )

    return warped_image, target_image
def save_model_weights(mdl):
    encoder.save_weights( mdl+"/encoder.h5"   )
    decoder_A.save_weights(mdl+"/decoder_A.h5" )
    decoder_B.save_weights( mdl+"/decoder_B.h5" )
    print( "Savign model weights" )

def loadImgs(path):
    #all_images = ( cv2.imread(fn) for fn in path )
    filenames = getfiles(path)
    all_images = [cv2.imread(img) for img in filenames]
    for i,imgs in enumerate(all_images):
        if i == 0:
            images = numpy.ndarray(shape=(len(filenames),256,256,3) , dtype=imgs.dtype )
        images[i] = imgs
    return images
def get_training_data( images, batch_size ):
    indices = numpy.random.randint( len(images), size=batch_size )
    for i,index in enumerate(indices):
        image = images[index]
        
        image = random_transform( image, **random_transform_args )

        warped_img, target_img = random_warp( image )
        #print(warped_img.shape)
        # first index => initiate array
        if i == 0:
            warped_images = numpy.empty( (batch_size,) + warped_img.shape, warped_img.dtype )
            target_images = numpy.empty( (batch_size,) + target_img.shape, warped_img.dtype )
            
        warped_images[i] = warped_img
        target_images[i] = target_img

    return warped_images, target_images

def Train(Odir,TAdir,Mdir,ep,sv):
    imgsA=loadImgs(Odir)/255.0
    imgsB=loadImgs(TAdir)/255.0
    imgsA += imgsB.mean(axis=(0,1,2))-imgsA.mean(axis=(0,1,2))
    try:
        encoder.load_weights( Mdir+"/encoder.h5"   )
        decoder_A.load_weights( Mdir+"/decoder_A.h5" )
        decoder_B.load_weights( Mdir+"/decoder_B.h5" )
        print( "loaded existing model" )
    except:
        print("No existing model" )
  
    for epoch in range(int(ep)):
        # get next training batch
        batch_size = 64
        warped_A, target_A = get_training_data( imgsA, batch_size )
        warped_B, target_B = get_training_data( imgsB, batch_size )

        # train and calculate loss
        loss_A = autoencoder_A.train_on_batch( warped_A, target_A )
        loss_B = autoencoder_B.train_on_batch( warped_B, target_B )

        if epoch % int(sv) == 0:
            print ("Training loss "+str(epoch)+" :")
            print( loss_A, loss_B )

            # save model every 100 steps
            save_model_weights(Mdir)
            test_A = target_A[0:14]
            test_B = target_B[0:14]
            # create image and write to disk
            
    # save our model after training has finished
    save_model_weights(Mdir)
