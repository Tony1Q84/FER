from keras.layers import Input, Dense
from keras.models import Model
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, concatenate, average
from keras.layers import Conv1D, MaxPool1D, GlobalAveragePooling1D
from keras.layers import SeparableConv2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.regularizers import l2

def mini_XCEPTION(input_shape):
    regularization = l2(0.01)

    # base
    img_input = Input(input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 4
    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = Conv2D(16, (3, 3),
               # kernel_regularizer=regularization,
               padding='same')(x)
    # x = GlobalAveragePooling2D()(x)
    # output = Activation('softmax', name='predictions')(x)
    # fc = Dense(64, activation='relu', kernel_regularizer=regularization)(x)

    model = Model(img_input, x)
    return model

# def tiny_XCEPTION(in)

def coor_covnet(input_shape):
    regularization = l2(0.01)

    coor_input = Input(input_shape)
    x = Conv1D(100, 3, activation='relu')(coor_input)
    x = Conv1D(100, 3, activation='relu')(x)
    x = MaxPool1D(3)(x)

    x = Conv1D(160, 3, activation='relu')(x)
    # x = Conv1D(160, 3, activation='relu')(x)
    x = MaxPool1D(3)(x)
    x = Flatten()(x)
    x = Dense(16, activation='softmax')(x)

    model = Model(coor_input, x)
    return model


def two_stream(img_input_shape, coor_input_shape, num_classes, l2_regularization=0.001):
    regularization = l2(l2_regularization)
    img_input = Input(shape=img_input_shape, name = 'img_input')
    coor_input = Input(shape=coor_input_shape, name = 'coor_input')

    img_net = mini_XCEPTION(img_input_shape)
    coor_net = coor_covnet(coor_input_shape)

    img_stream = img_net(img_input)
    coor_stream = coor_net(coor_input)

    img_stream = Flatten()(img_stream)
    # coor_stream = Flatten()(coor_stream)

    img_stream = Dense(14, activation='relu', kernel_regularizer=regularization)(img_stream)
    coor_stream = Dense(14, activation='relu', kernel_regularizer=regularization)(coor_stream)

    feature = average([img_stream, coor_stream])
    fc_1 = Dense(units=num_classes, activation='relu', kernel_regularizer=regularization)(feature)
    output = Activation('softmax', name = 'predictions')(fc_1)

    model = Model(inputs=[img_input, coor_input], outputs=output)
    return model