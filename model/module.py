import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,Dropout, Flatten, Dense
from tensorflow.keras.utils import plot_model

class CNN(Model):
    def __init__(self, input_shape, in_channel, num_class):
        super(CNN, self).__init__()
        self.c1 = Conv2D(3*in_channel, (3, 3), activation='relu', input_shape=input_shape)
        self.c2 = Conv2D(9*in_channel, (3, 3), activation='relu')
        self.d1 = Dropout(0.25)
        self.flatten = Flatten()
        self.f1 = Dense(6*in_channel, activation='relu')
        self.d2 = Dropout(0.5)
        self.f2 = Dense(num_class, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.d1(x)
        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        return y

