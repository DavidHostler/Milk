import tensorflow
import tensorflow as tf 
from tensorflow.keras.layers import Dense,  Conv2D, MaxPooling2D, UpSampling2D, Dropout, Flatten,LeakyReLU
from tensorflow.keras.layers import  BatchNormalization,PReLU, Model, layers, add
from tensorflow.keras.models import Sequential



class Generator:
    def __init__(self):
        self.generator_model = None
        self.shared_axes = [1,2]
        self.no_resblocks = 16

    def residual_block(self, input):
        
        model = Sequential()

        model.add(Conv2D(64, (3,3), padding = "same")(input)) 
        model.add(BatchNormalization(momentum = 0.6)(model)) 
        model.add(PReLU(shared_axes = self.shared_axes)(model)) 
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


    def discriminator_block(self, inputs, filters, strides=1, batchnorm=True):
        disc_model = Sequential()
        disc_model.add(Conv2D(filters, (3,3), strides = strides, padding="same")(inputs))
        disc_model.add(LeakyReLU( alpha=0.2 )(disc_model))
        if batchnorm:
            disc_model.add(BatchNormalization( momentum=0.6 )(disc_model))

        
        return disc_model



    def create_discriminator(self,disc_inputs):

        df = 64
        d = Sequential()
        d.add(discriminator_block(disc_inputs, df, batchnorm=False))
        d.add(discriminator_block(d, df, strides=2))
        d.add(discriminator_block(d, df*2))
        d.add(discriminator_block(d, df*2, strides=2))
        d.add(discriminator_block(d, df*4))
        d.add(discriminator_block(d, df*4, strides=2))
        d.add(discriminator_block(d, df*8))
        d.add(discriminator_block(d, df*8, strides=2))
        
        d8_5 = Flatten()(d)
        d8_5.add(Dense(df*16)(d8_5))
        d8_5.add(LeakyReLU(alpha=0.2)(d8_5))
        #Might want to experiment with different activation functions here for the output
        validity = Sequential()
        validity.add(Dense(1, activation='sigmoid')(d))

        return Model(disc_inputs, validity)