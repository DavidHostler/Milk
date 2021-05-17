import tensorflow
import tensorflow as tf 
from tensorflow.keras.layers import Dense,  Conv2D, MaxPooling2D, Dropout, Flatten 
from tensorflow.keras.models import Sequential  
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
 
class Generator:
    def __init__(self):
        self.generator_model = None
        self.discriminator_model = None


    def residual_block(self, input):
        
        model = Sequential()

        model.add(Conv2D(64, (3,3), padding = "same")(input))
        #model = Conv2D(64, (3,3), padding = "same")(input)
        model.add(BatchNormalization(momentum = 0.6)(model))
        #model = BatchNormalization(momentum = 0.5)(model)
        model.add(PReLU(shared_axes = [1,2])(model))
        #model = PReLU(shared_axes = [1,2])(model)
        model.add(Conv2D(64, (3,3), padding = "same")(model))
        #model = Conv2D(64, (3,3), padding = "same")(model)
        model.add(BatchNormalization(momentum = 0.6)(model))
        #model = BatchNormalization(momentum = 0.5)(model)
    
        return add([input,model])

    def B_block(self, input):

        model = Sequential()
        
        model.add(Conv2D(64, (3,3), padding = "same")(input))
        #model = Conv2D(256, (3,3), padding="same")(input)
        model.add(UpSampling2D( size = 2 )(model))(input)
        #model = UpSampling2D( size = 2 )(model)
        model.add(PReLU(shared_axes=[1,2])(model))
        #model = PReLU(shared_axes=[1,2])(model)
    
        return model
    
    def create_gen(self, generator_inputs):
        model = Sequential()
        model.add(Conv2D(64, (9,9), padding = "same"))
        #model = Conv2D(64, (9,9), padding="same")(generator_inputs)
        #model = PReLU(shared_axes=[1,2])(model)
        model.add(PReLU(shared_axes=[1,2])(model))
        temp = model

        for i in range(num_residual_block):
            #model = residual_block(model)
            model.add(PReLU(shared_axes=[1,2])(model))

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



    def create_disc(disc_inputs):

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
        validity.add(dense(1, activation='sigmoid')(d10))

        return Model(disc_inputs, validity)

