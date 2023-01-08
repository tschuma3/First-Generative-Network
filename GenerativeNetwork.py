"""
Help from
-- https://anderfernandez.com/en/blog/how-to-code-gan-in-python/
"""


import keras
import tensorflow as tf
from keras.layers import Dense, Conv2DTranspose, LeakyReLU, Reshape, BatchNormalization, Activation, Conv2D, Flatten, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.datasets import cifar10
import matplotlib.pyplot as plt
#%matplotlib inline -> Uncomment if using Jupyter
import numpy as np
import random
import pandas as pd
from datetime import datetime

#region Generate Images

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

    return discriminator

model_Discriminator = discriminator_Images()
model_Discriminator.summary()

#endregion

#region Load Data from Cifar10

def carry_Images():
    (Xtrain, Ytrain), (_,_) = cifar10.load_data()

    indice = np.where(Ytrain == 0)
    indice = indice[0]
    Xtrain = Xtrain[indice, :, :, :]

    X = Xtrain.astype('float32')
    X = (X - 127.5) / 127.5

    return X

print(carry_Images().shape)

#endregion

#region Training

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