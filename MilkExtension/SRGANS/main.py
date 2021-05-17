from SRGANS.src.model import Generator, Discriminator
from SRGANS.src.data import DataGenerator
import os
import cv2
import numpy as np



import matplotlib.pyplot as plt
from keras import Model, layers
from keras.applications import VGG19
generator_object= Generator()
discriminator_object = Discriminator()
 


Input = layers.Input  
data_gen = DataGenerator()
train_low_res_images, train_high_res_images = data_gen.get_training_data()
test_low_res_images, test_high_res_images = data_gen.get_testing_data()
num_res_block = generator_object.no_resblocks
high_res_shape = (train_high_res_images.shape[1], train_high_res_images.shape[2], train_high_res_images.shape[3])
low_res_shape = (train_low_res_images.shape[1], train_low_res_images.shape[2], train_low_res_images.shape[3])

low_res_inputs = Input(shape=low_res_shape)
high_res_inputs = Input(shape=high_res_shape)
generator = generator_object.create_gen(low_res_inputs)

discriminator = discriminator_object.create_disc(high_res_inputs)

discriminator.compile(loss = "binary_crossentropy", optimizer="adam", metrics=['accuracy'])

#Build vgg pretrained model weights
def build_vgg():
    vgg = VGG19(weights="imagenet")
    vgg.outputs = [vgg.layers[9].output]

    img = Input(shape=high_res_shape)

    img_features = vgg(img)

    return Model(img, img_features)


vgg = build_vgg()
vgg.trainable = False

#Used to combine the Discriminator and Generator into GAN model

def create_comb(gen_model, disc_model, vgg, lr_ip, hr_ip):
    gen_img = gen_model(lr_ip)
    
    gen_features = vgg(gen_img)
    
    disc_model.trainable = False
    validity = disc_model(gen_img)
    
    return Model(inputs=[lr_ip, hr_ip], outputs=[validity, gen_features])


gan_model = create_comb(generator, discriminator, vgg, low_res_inputs, high_res_inputs)
gan_model.compile(loss=["binary_crossentropy","mse"], loss_weights=[1e-3, 1], optimizer="adam") 



 

def train(batch_size, epochs):
    #epochs = 100
    train_low_res_images_batches = []
    train_high_res_images_batches = []
    for it in range(int(train_high_res_images.shape[0] / batch_size)):
        start_idx = it * batch_size
        end_idx = start_idx + batch_size
        train_high_res_images_batches.append(train_high_res_images[start_idx:end_idx])
        train_low_res_images_batches.append(train_low_res_images[start_idx:end_idx])

    for e in range(epochs):
        
        gen_label = np.zeros((batch_size, 1))
        real_label = np.ones((batch_size,1))
        g_losses = []
        d_losses = []
        for b in range(len(train_high_res_images_batches)):
            lr_imgs = train_low_res_images_batches[b]
            hr_imgs = train_high_res_images_batches[b]
            
            gen_imgs = generator.predict_on_batch(lr_imgs)
            
            discriminator.trainable = True
            d_loss_gen = discriminator.train_on_batch(gen_imgs, gen_label)
            d_loss_real = discriminator.train_on_batch(hr_imgs, real_label)
            discriminator.trainable = False
            
            d_loss = 0.5 * np.add(d_loss_gen, d_loss_real) 
            
            image_features = vgg.predict(hr_imgs)

            
            g_loss, _, _ = gan_model.train_on_batch([lr_imgs, hr_imgs], [real_label, image_features])
            
            d_losses.append(d_loss)
            g_losses.append(g_loss)
            
        g_losses = np.array(g_losses)
        d_losses = np.array(d_losses)
        
        g_loss = np.sum(g_losses, axis=0) / len(g_losses)
        d_loss = np.sum(d_losses, axis=0) / len(d_losses)
        
        print("epoch:", e+1 ,"g_loss:", g_loss, "d_loss:", d_loss)
        