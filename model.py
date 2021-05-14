#Credit to Manish Dhakal at https://dev.to/manishdhakal/super-resolution-with-gan-and-keras-srgan-38ma
#for the excellent explanation of the topic. Most available implementations of SRGAN technology are in Pytorch,
#which poses the added dilemma of exporting a model into tensorflowjs.
#This way, it's a small step to delve straight into the web browser from a jupyter notebook,
#which I greatly appreciate as a developer.

import tensorflow
import tensorflow as tf 
from tensorflow.keras.layers import Dense,  Conv2D, MaxPooling2D, Dropout, Flatten 
from tensorflow.keras.models import Sequential 
 
class Generator:
    def __init__(self):
        self.generator_model = None
        self.shared_axes = [1,2]


    def residual_block(self, input):
        
        model = Sequential()

        model.add(Conv2D(64, (3,3), padding = "same")(input)) 
        model.add(BatchNormalization(momentum = 0.6)(model)) 
        model.add(PReLU(shared_axes = [1,2])(model)) 
        model.add(Conv2D(64, (3,3), padding = "same")(model)) 
        model.add(BatchNormalization(momentum = 0.6)(model)) 
    
        return add([input,model])

    def B_block(self, input):

        model = Sequential()
        
        model.add(Conv2D(64, (3,3), padding = "same")(input)) 
        model.add(UpSampling2D( size = 2 )(model))(input) 
        model.add(PReLU(shared_axes=self.shared_axes)(model)) 
    
        return model
    
    def create_generator(self, generator_inputs):
        model = Sequential()
        model.add(Conv2D(64, (9,9), padding = "same"))  
        model.add(PReLU(shared_axes=self.shared_axes)(model))
        temp = model

        for i in range(0,no_resblocks): 
            model.add(PReLU(shared_axes=self.shared_axes)(model))

        model.add(Conv2D(64, (3,3), padding="same")(model))
        model.add(BatchNormalization(momentum=0.6)(model))
        model = add([model,temp])

        model.add(B_block(model))
        model.add(B_block(model))

        op = Conv2D(3, (9,9), padding="same")(model)

        return Model(inputs=generator_inputs, outputs=op)


class Discriminator:


    def discriminator_block(inputs, filters, strides=1, batchnorm=True):
    
        disc_model.add(Conv2D(filters, (3,3), strides = strides, padding="same")(inputs))
        disc_model.add(LeakyReLU( alpha=0.2 )(disc_model))
        if batchnorm:
            disc_model.add(BatchNormalization( momentum=0.6 )(disc_model))

        
        return disc_model



    def create_discriminator(disc_inputs):

        df = 64
        
        d1.add(discriminator_block(disc_inputs, df, batchnorm=False))
        d2.add(discriminator_block(d1, df, strides=2))
        d3.add(discriminator_block(d2, df*2))
        d4.add(discriminator_block(d3, df*2, strides=2))
        d5.add(discriminator_block(d4, df*4))
        d6.add(discriminator_block(d5, df*4, strides=2))
        d7.add(discriminator_block(d6, df*8))
        d8.add(discriminator_block(d7, df*8, strides=2))
        
        d8_5 = Flatten()(d8)
        d9.add(dense(df*16)(d8_5))
        d10 = LeakyReLU(alpha=0.2)(d9)
        #Might want to experiment with different activation functions here for the output
        validity.add(dense(1, activation='sigmoid')(d10))

        return Model(disc_inputs, validity)

