from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D, Activation
from tensorflow.keras.models import Sequential

class ConvNetFactory:
    def __init__(self):
        pass

    @staticmethod
    def build(name, *args, **kargs):
        mappings = {
            "shallownet": ConvNetFactory.ShallowNet,
            "lenet": ConvNetFactory.LeNet,
            "karpathynet" : ConvNetFactory.KarpathyNet,
            "minivggnet" : ConvNetFactory.MiniVGGNet
        }

        builder = mappings.get(name, None)
        if builder is None:
            return None
        
        return builder(*args, **kargs)

    #Shallow Net Input=>Conv=>RELU=>FC
    @staticmethod
    def ShallowNet(numChannels, imgRows, imgCols, numClasses, **kargs):
        model = Sequential()
        inputShape = (imgRows, imgCols, numChannels)

        model.add(Conv2D(32, (3,3), padding = 'same', input_shape = inputShape))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(numClasses))
        model.add(Activation('softmax'))

        return model

    @staticmethod
    def LeNet(numChannels, imgRows, imgCols, numClasses, activation = 'tanh', **kargs):
        model = Sequential()
        inputShape = (imgRows, imgCols, numChannels)

        model.add(Conv2D(20,(5,5), padding = 'same', input_shape = inputShape))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

        model.add(Conv2D(50,(5,5), padding = 'same', input_shape = inputShape))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation))

        model.add(Dense(numClasses))

        model.add(Activation('softmax'))

        return model

    @staticmethod
    def KarpathyNet(numChannels, imgRows, imgCols, numClasses, dropout = False, **kwargs):
        model = Sequential()
        inputshape = (imgRows, imgCols, numChannels)

        model.add(Conv2D(16, (5,5), padding = 'same', input_shape = inputshape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

        if dropout:
            model.add(Dropout(0.25))

        #CONV->RELU->POOL
        model.add(Conv2D(32, (5,5), padding = 'same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

        if dropout:
            model.add(Dropout(0.25))

        #CONV->RELU->POOL
        model.add(Conv2D(64, (5,5), padding = 'same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

        if dropout:
            model.add(Dropout(0.5))

        #FC->RELU
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))

        if dropout:
            model.add(Dropout(0.5))

        #FC->SM
        model.add(Dense(numClasses))
        model.add(Activation('softmax'))

        return model


    @staticmethod
    def MiniVGGNet(numChannels, imgRows, imgCols, numClasses, dropout = False, **kwargs):
        model = Sequential()
        inputshape = (imgRows, imgCols, numChannels)
        chanDims = -1

        model.add(Conv2D(32, (3,3), padding = 'same', input_shape = inputshape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanDims))
        model.add(Conv2D(32, (3,3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

        if dropout:
            model.add(Dropout(0.25))

        model.add(Conv2D(64, (3,3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanDims))
        model.add(Conv2D(64, (3,3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

        if dropout:
            model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanDims))

        if dropout:
            model.add(Dropout(0.5))

        model.add(Dense(numClasses))
        model.add(Activation('softmax'))

        return model


