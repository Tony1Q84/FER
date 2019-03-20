from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout, SeparableConv2D, MaxPooling2D
from keras.layers import Activation, BatchNormalization, add, Reshape
from keras.applications.mobilenet import relu6, DepthwiseConv2D
from keras.utils.vis_utils import plot_model
from keras.regularizers import l2
from keras import backend as K
from keras import layers

def _conv_block(inputs, filters, kernel, strides, l2_regularization=0.01):
    """Convolution Block
        This function defines a 2D convolution operation with BN and relu6.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.

    # Returns
        Output tensor.
    """
    regularization = l2(l2_regularization)

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides, kernel_regularizer=regularization)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation(relu6)(x)

def _bottleneck(inputs, filters, kernel, t, s, r = False, l2_regularization=0.01):
    """Bottleneck
       This function defines a basic bottleneck structure.

   # Arguments
       inputs: Tensor, input tensor of conv layer.
       filters: Integer, the dimensionality of the output space.
       kernel: An integer or tuple/list of 2 integers, specifying the
           width and height of the 2D convolution window.
       t: Integer, expansion factor.
           t is always applied to the input size.
       s: An integer or tuple/list of 2 integers,specifying the strides
           of the convolution along the width and height.Can be a single
           integer to specify the same value for all spatial dimensions.
       r: Boolean, Whether to use the residuals.

   # Returns
       Output tensor.
   """
    regularization = l2(l2_regularization)

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularization)(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = add([x, inputs])

    return x

def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    """Inverted Residual Block
        This function defines a sequence of 1 or more identical layers.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.

    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)

    return x

def MobileNetV2(input_shape, num_classes):
    # 2284615
    """MobileNetv2
        This function defines a MobileNetv2 architectures.
    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        k: Integer, number of classes.
    # Returns
        MobileNetv2 model.
    """

    inputs = Input(shape=input_shape)
    x = _conv_block(inputs, 32, (3, 3), strides=(2, 2))

    x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=1)
    x = _inverted_residual_block(x, 24, (3, 3), t=6, strides=2, n=2)
    x = _inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=4)
    x = _inverted_residual_block(x, 96, (3, 3), t=6, strides=1, n=3)
    x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)

    x = _conv_block(x, 1280, (1, 1), strides=(1, 1))
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1280))(x)
    x = Dropout(0.3, name='Dropout')(x)
    x = Conv2D(num_classes, (1, 1), padding='same')(x)

    x = Activation('softmax', name = 'softmax')(x)
    output = Reshape((num_classes,))(x)

    model = Model(inputs, output)
    # plot_model(model, to_file='../images/MobileNetv2.png', show_shapes=True)

    return model

def MobileNetV3(input_shape, num_classes):
    # 121183
    # 69583
    # 66079
    """MobileNetv3
        This function defines a MobileNetv3 architectures.
    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        k: Integer, number of classes.
    # Returns
        MobileNetv3 model.
    """

    inputs = Input(shape=input_shape)
    x = _conv_block(inputs, 8, (3, 3), strides=(1, 1))

    x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=2, n=1)
    x = _inverted_residual_block(x, 24, (3, 3), t=6, strides=2, n=2)
    x = _inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=2)
    x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=1)
    x = Dropout(0.3, name='Dropout')(x)

    x = Conv2D(num_classes, (3, 3), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name = 'softmax')(x)

    model = Model(inputs, output)

    return model

def MobileNetV4(input_shape, num_classes):
    # 187015
    """MobileNetv4
        This function defines a MobileNetv4 architectures.
    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        k: Integer, number of classes.
    # Returns
        MobileNetv4 model.
    """

    inputs = Input(shape=input_shape)
    x = _conv_block(inputs, 8, (3, 3), strides=(1, 1))
    # x = _conv_block(x, 8, (3, 3), strides=(1, 1))

    x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=1)
    x = _inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=2)
    x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=2)
    x = _inverted_residual_block(x, 128, (3, 3), t=6, strides=2, n=1)

    x = Dropout(0.3, name='Dropout')(x)
    x = Conv2D(num_classes, (3, 3), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name = 'softmax')(x)

    model = Model(inputs, output)

    return model

def self_model(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)

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

    x = _bottleneck(x, 16, (3, 3), t=6, s=1)
    x = _bottleneck(x, 16, (3, 3), t=6, s=1)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = _bottleneck(x, 32, (3, 3), t=6, s=1)
    x = _bottleneck(x, 32, (3, 3), t=6, s=1)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = _bottleneck(x, 64, (3, 3), t=6, s=1)
    x = _bottleneck(x, 64, (3, 3), t=6, s=1)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 4
    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = _bottleneck(x, 128, (3, 3), t=6, s=1)
    x = _bottleneck(x, 128, (3, 3), t=6, s=1)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = Conv2D(num_classes, (3, 3),
               # kernel_regularizer=regularization,
               padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)

    model = Model(img_input, output)
    # plot_model(model, to_file='../images/mini_XCEPTION.png', show_shapes=True)
    return model

