import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Conv2DTranspose, UpSampling2D, Activation
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import os
from pixel_shuffler import PixelShuffler

IMAGE_SHAPE = (128,128,3)
kernel_size = 5
optimizer = Adam( lr=5e-5, beta_1=0.5, beta_2=0.999 )

def encoder():
    inputs=Input(shape=IMAGE_SHAPE)
    x = inputs
    ##x = K.constant(inp)
    #x = Conv2D( 64, kernel_size=5, strides=2, padding='same' )(x)
    #x = LeakyReLU(0.1)(x)
    x = Conv2D( 128, kernel_size=5, strides=2, padding='same' )(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D( 256, kernel_size=5, strides=2, padding='same' )(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D( 512, kernel_size=5, strides=2, padding='same' )(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D( 1024, kernel_size=5, strides=2, padding='same' )(x)
    x = LeakyReLU(0.1)(x)
    x = Dense( 512 )( Flatten()(x) )
    x = Dense(8*8*512)(x)
    x = Reshape((8,8,512))(x)
    
    #arr=K.eval(x)
    #return arr
    return Model(inputs,x)
def decoder():
    #x = K.constant(inp)
    inputs = Input( shape=(8,8,512))
    x = inputs
    x = Conv2D( 1024, kernel_size=3, padding='same' )(x)
    x = LeakyReLU(0.1)(x)
    x = PixelShuffler()(x)
    x = Conv2D( 128*4, kernel_size=3, padding='same' )(x)
    x = LeakyReLU(0.1)(x)
    x = PixelShuffler()(x)
    x = Conv2D( 64*4, kernel_size=3, padding='same' )(x)
    x = LeakyReLU(0.1)(x)
    x = PixelShuffler()(x)
    x = Conv2D( 32*4, kernel_size=3, padding='same' )(x)
    x = LeakyReLU(0.1)(x)
    x = PixelShuffler()(x)
    #x = Conv2D( 16*4, kernel_size=3, padding='same' )(x)
    #x = LeakyReLU(0.1)(x)
    #x = PixelShuffler()(x)
    x = Conv2D( 3, kernel_size=5, padding='same',activation="sigmoid" )(x)
    return Model(inputs,x)

encoder = encoder()
decoder_A = decoder()
decoder_B = decoder()
x = Input( shape=IMAGE_SHAPE )
autoencoder_A = Model( x, decoder_A( encoder(x) ) )
autoencoder_B = Model( x, decoder_B( encoder(x) ) )
autoencoder_A.compile( optimizer=optimizer, loss='mean_absolute_error' )
autoencoder_B.compile( optimizer=optimizer, loss='mean_absolute_error' )
print(autoencoder_A.summary())
