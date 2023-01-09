"""
Help from
-- https://anderfernandez.com/en/blog/how-to-code-gan-in-python/
"""

"""
Tips for Generative Adversarial Network
1. Generate one type of image 
    -- At the beginning try and generate one thing, e.g. dogs, planes, faces
2. Fail quick and improve
    -- Letting the model train for 20 to 30 epochs instead of 100 to 200
3. Identfy the metric to evaluate your model
    -- Normal neural network have quite clear indicators when visualizing
    -- In this case the program focuses on accuracy 
4. If the session ends...load your model
    -- If the session ends then the program can load the last trained model avoiding to retrain from scratch
"""

import keras
import tensorflow as tf
from keras.layers import Dense, Conv2DTranspose, LeakyReLU, Reshape, BatchNormalization, Activation, Conv2D, Flatten, Dropout
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.datasets import cifar10
import matplotlib.pyplot as plt
#%matplotlib inline -> Uncomment if using Jupyter
import numpy as np
import random
import pandas as pd
from datetime import datetime

"""
Libraries
    -- Keras
        -- https://keras.io/
            -- Conv2DTranspose
                -- https://keras.io/api/layers/convolution_layers/convolution2d_transpose/
            -- Conv2D
                -- https://www.geeksforgeeks.org/keras-conv2d-class/
            -- Dense
                -- https://keras.io/api/layers/core_layers/dense/
            -- LeakyReLU
                -- https://keras.io/api/layers/activation_layers/leaky_relu/
"""


#region Load the Last Saved Model

model_Generator = load_model('D:\GitHub Repos\First-Generative-Network\Second Run\129_20230108_220337_model_generator_.h5')

#endregion

#region Generate Images

"""
-- This network generates images
-- In the beginning the network will only generate noise 

Coding the Generative Network
-- Dense: The noise layer of the generator
-- Conv2DTranspose: Enables convolve to upscale and convolve the image at the same time
    -- Similar to using the function UpSampling2D followed by Conv2D
-- LeakyReLU: Better than ReLU function as it avoids gradient vanish
-- BatchNormalization: Enables the normalization of the results of a convolution
    -- Allows for better results in some cases
Reshape: Enables the transformation of a one-dimensional vector into a three-dimensional array
"""


def generate_Images():
    generator = Sequential()

    generator.add(Dense(256 * 4 * 4, input_shape = (100,)))
    #generator.add(BatchNormalization())
    generator.add(LeakyReLU())
    generator.add(Reshape((4, 4, 256)))

    generator.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"))
    #generator.add(BatchNormalization())
    generator.add(LeakyReLU(alpha=0.2))

    generator.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"))
    #generator.add(BatchNormalization())
    generator.add(LeakyReLU(alpha=0.2))

    generator.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"))
    #generator.add(BatchNormalization())
    generator.add(LeakyReLU(alpha=0.2))

    generator.add(Conv2D(3, kernel_size=3, padding="same", activation='tanh'))

    return generator

model_Generator = generate_Images()

model_Generator.summary()

#endregion

#region Draw the GAN

"""
-- At this point in the code it will generate random images as it is not yet trained
-- Will only generate noise
"""


def generate_Input_Data(n_Samples):
    X = np.random.randn(100 * n_Samples)
    X =X.reshape(n_Samples, 100)
    return X

def create_Fake(model_Generator, n_Samples):
    input = generate_Input_Data(n_Samples)
    X = model_Generator.predict(input)
    y = np.zeros((n_Samples, 1))
    return X, y

samples = 4
X,_ = create_Fake(model_Generator, samples)

#Vizualize the results
for i in range(samples):
    plt.subplot(2, 2, 1 + i)
    plt.axis('off')
    plt.imshow(X[i])

#endregion

#region Discriminator Network

"""
-- This network classifies whether an image is real or not
-- This is the network that will enable the generative network to be trained
-- Takes an image input and outputs a binary value
-- Dropout during each convolution helps avoid overfitting
-- The sigmoid activation function helps determine the probibility of an image being in the target group
"""


def discriminator_Images():

    discriminator = Sequential()

    discriminator.add(Conv2D(64, kernel_size=3, padding="same", input_shape=(32, 32, 3)))
    discriminator.add(LeakyReLU(alpha=0.2))
    #discriminator.add(Dropout(0.2))

    discriminator.add(Conv2D(128, kernel_size=3, strides=(2,2), padding="same"))
    discriminator.add(LeakyReLU(alpha=0.2))
    #discriminator.add(Dropout(0.2))

    discriminator.add(Conv2D(128, kernel_size=3, strides=(2,2), padding="same"))
    discriminator.add(LeakyReLU(alpha=0.2))
    #discriminator.add(Dropout(0.2))

    discriminator.add(Conv2D(256, kernel_size=3, strides=(2,2), padding="same"))
    discriminator.add(LeakyReLU(alpha=0.2))
    #discriminator.add(Dropout(0.2))

    discriminator.add(Flatten())
    discriminator.add(Dropout(0.4))
    discriminator.add(Dense(1, activation='sigmoid'))

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    #Loss and binary cross entropy ---> https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a

    return discriminator

model_Discriminator = discriminator_Images()
model_Discriminator.summary()

#endregion

#region Load Data from Cifar10

"""
-- This is a dataset of 50000 32 X 32 images used to train ai
-- Want to normalize the data, which allows the model to work faster
    -- The RGB layer goes from 0 to 255
    -- Both subtract and divide by 127.5 allowing the data to go from -1 to 1
"""


def carry_Images():

    #Change this to another dataset to train the model with a different dataset
    #Other possible datasets: https://archive.ics.uci.edu/ml/index.php
    (Xtrain, Ytrain), (_,_) = cifar10.load_data() #<---- cifar10 to switch what dataset the model trains with

    #This allows to change what the model imitates
    indice = np.where(Ytrain == 0) #<---- 0 to switch what image the model trains with

    indice = indice[0]
    Xtrain = Xtrain[indice, :, :, :]

    X = Xtrain.astype('float32')
    X = (X - 127.5) / 127.5

    return X

print(carry_Images().shape)

#endregion

#region Training

"""
-- Creates a function that generates both real and fake images
-- Generate fake images with similar size to the real ones, 32 X 32 X 3
-- Give a label of 0 to the fake images to better improve performance
"""


def carry_Data_Real(dataset, n_Samples):
    ix = np.random.randint(0, dataset.shape[0], n_Samples)
    X = dataset[ix]
    y = np.ones((n_Samples, 1))
    return X, y

def carry_Data_Fake(n_Samples):
    X = np.random.rand(32 * 32 * 3 * n_Samples)
    X = -1 + X * 2
    X = X.reshape((n_Samples, 32, 32, 3))
    y = np.zeros((n_Samples, 1))
    return X, y

"""
-- It is important to pre-train the discriminator because when training the GAN, it will only train the generator
-- To see if the discriminator has been trained correctly, it iwill check the accuracy of the real and fake data
-- To train it, passing half of the real data and half of the fake data
    -- Hence it calculates the mid batch
"""

def train_Discriminator(model, dataset, n_Iterations=20, batch=128):
    
    mid_Batch = int(batch / 2)

    for i in range(n_Iterations):
        X_Real, y_Real = carry_Data_Real(dataset, mid_Batch)
        _, acc_Real = model.train_on_batch(X_Real, y_Real)

        X_Fake, y_Fakes = carry_Data_Fake(mid_Batch)
        _, acc_Fake = model.train_on_batch(X_Fake, y_Fakes)

        #print(str(i + 1) + 'Real: ' + str(acc_Real * 100) + ' , Fake: ' + str(acc_Fake * 100))
        print(f"{i + 1}. Real: {str(acc_Real * 100)}, Fake: {str(acc_Fake * 100)}")

dataset = carry_Images()
train_Discriminator(model_Discriminator, dataset)

#endregion

#region The Generative Adversarial Network

"""
-- Will connect the generator network and discriminator netowrk
-- The trainable parameter will be set to false because the discriminator is already trained
-- The cost function will be 'binary_crossentropy', as it will help classify between fake (0) and real (1) images
"""

def create_GAN(discriminator, generator):
    discriminator.trainable=False
    gan = Sequential()
    gan.add(generator)
    gan.add(discriminator)

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    gan.compile(loss="binary_crossentropy", optimizer=opt)

    return gan

gan = create_GAN(model_Discriminator, model_Generator)
gan.summary()

#endregion

#region Model Evaluation and Image Generation

"""
-- Able to see the weights and the images generated, showing the improvement
-- Starts with saving 10 of the results
-- Saves the model as it gets trained
-- Generate new data to evaluate the model
    -- It is better to train with new data then with the ones just trained with
"""


def show_Images_Generator(data_Fake, epoch):

    now = datetime.now()
    now = now.strftime("%Y%m%d_%H%M%S")

    data_Fake = (data_Fake + 1) / 2.0

    for i in range(10):
        plt.imshow(data_Fake[i])
        plt.axis('off')
        number = str(epoch) + '_image_generator_' + str(i) + '.png'
        plt.savefig(number, bbox_inches='tight')
        plt.close()

def evaluate_And_Save(model_Generator, epoch, mid_Dataset):

    #Saves the data
    now = datetime.now()
    now = now.strftime("%Y%m%d_%H%M%S")
    number = str(epoch) + '_' + str(now) + "_model_generator_" + '.h5'
    model_Generator.save(number)

    #Generates new data
    X_Real, Y_Real = carry_Data_Real(dataset, mid_Dataset)
    X_Fake, Y_Fake = create_Fake(model_Generator, mid_Dataset)

    _, acc_Real = model_Discriminator.evaluate(X_Real, Y_Real)
    _, acc_Fake = model_Discriminator.evaluate(X_Fake, Y_Fake)

    print(f"Acc Real: {str(acc_Real * 100)}% Acc Fake: {str(acc_Fake * 100)}%")

def training(data, model_Generator, model_Discriminator, epochs, n_Batch, beginning=0):

    dimension_Batch = int(data.shape[0] / n_Batch)
    mid_Dataset = int(n_Batch / 2)

    #Iterates over epoch
    for epoch in range(beginning, beginning + epochs): 
        
        #Iterates over the batches
        for batch in range(n_Batch):

            #Loads the real data
            X_Real, Y_Real = carry_Data_Real(dataset, mid_Dataset)

            #Train the discriminator with real data
            cost_Discriminator_Real, _ = model_Discriminator.train_on_batch(X_Real, Y_Real)

            #Create fake data and train discriminator with fake data
            X_Fake, Y_Fake = create_Fake(model_Generator, mid_Dataset)
            cost_Discriminator_Fake, _ = model_Discriminator.train_on_batch(X_Fake, Y_Fake)

            #Generate input images for the GAN
            X_Gan = generate_Input_Data(mid_Dataset)
            Y_Gan = np.ones((mid_Dataset, 1))

            #Train GAN with fake data
            cost_Gan = gan.train_on_batch(X_Gan, Y_Gan)

        #Show the results every 10 epochs
        if (epoch + 1) % 10 == 0:
            evaluate_And_Save(model_Generator, epoch = epoch, mid_Dataset = mid_Dataset)
            show_Images_Generator(X_Fake, epoch = epoch)

training(dataset, model_Generator, model_Discriminator, epochs=300, n_Batch=128, beginning=0)

X_Fake, _ = create_Fake(model_Generator = model_Generator, n_Samples=49)
X_Fake = (X_Fake + 1) / 2

for i in range(49):
    plt.subplot(7, 7, i + 1)
    plt.axis('off')
    plt.imshow(X_Fake[i])

#endregion