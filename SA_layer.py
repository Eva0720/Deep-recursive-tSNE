#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:23:27 2019

@author: zzhou2
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:57:35 2019

@author: zzhou2
"""

from keras.engine.network import Layer

from keras.layers import InputSpec

import keras.backend as K

import tensorflow as tf

from keras.engine import *

from keras.legacy import interfaces

from keras import activations

from keras import initializers

from keras import regularizers

from keras import constraints

from keras.utils.generic_utils import func_dump

from keras.utils.generic_utils import func_load

from keras.utils.generic_utils import deserialize_keras_object

from keras.utils.generic_utils import has_arg

from keras.utils import conv_utils

from keras.legacy import interfaces

from keras.layers import Dense, Conv1D, Conv2D, Conv3D, Conv2DTranspose, Embedding




class ConvSN2D(Conv2D):



    def build(self, input_shape):

        if self.data_format == 'channels_first':

            channel_axis = 1

        else:

            channel_axis = -1

        if input_shape[channel_axis] is None:

            raise ValueError('The channel dimension of the inputs '

                             'should be defined. Found `None`.')

        input_dim = input_shape[channel_axis]

        kernel_shape = self.kernel_size + (input_dim, self.filters)



        self.kernel = self.add_weight(shape=kernel_shape,

                                      initializer=self.kernel_initializer,

                                      name='kernel',

                                      regularizer=self.kernel_regularizer,

                                      constraint=self.kernel_constraint)



        if self.use_bias:

            self.bias = self.add_weight(shape=(self.filters,),

                                        initializer=self.bias_initializer,

                                        name='bias',

                                        regularizer=self.bias_regularizer,

                                        constraint=self.bias_constraint)

        else:

            self.bias = None

            

        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),

                         initializer=initializers.RandomNormal(0, 1),

                         name='sn',

                         trainable=False)

        

        # Set input spec.

        self.input_spec = InputSpec(ndim=self.rank + 2,

                                    axes={channel_axis: input_dim})

        self.built = True

    def call(self, inputs, training=None):

        def _l2normalize(v, eps=1e-12):

            return v / (K.sum(v ** 2) ** 0.5 + eps)

        def power_iteration(W, u):

            #Accroding the paper, we only need to do power iteration one time.

            _u = u

            _v = _l2normalize(K.dot(_u, K.transpose(W)))

            _u = _l2normalize(K.dot(_v, W))

            return _u, _v

        #Spectral Normalization

        W_shape = self.kernel.shape.as_list()

        #Flatten the Tensor

        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])

        _u, _v = power_iteration(W_reshaped, self.u)

        #Calculate Sigma

        sigma=K.dot(_v, W_reshaped)

        sigma=K.dot(sigma, K.transpose(_u))

        #normalize it

        W_bar = W_reshaped / sigma

        #reshape weight tensor

        if training in {0, False}:

            W_bar = K.reshape(W_bar, W_shape)

        else:

            with tf.control_dependencies([self.u.assign(_u)]):

                W_bar = K.reshape(W_bar, W_shape)

                

        outputs = K.conv2d(

                inputs,

                W_bar,

                strides=self.strides,

                padding=self.padding,

                data_format=self.data_format,

                dilation_rate=self.dilation_rate)

        if self.use_bias:

            outputs = K.bias_add(

                outputs,

                self.bias,

                data_format=self.data_format)

        if self.activation is not None:

            return self.activation(outputs)

        return outputs



class SelfAttention(Layer):



    def __init__(self, ch=32, **kwargs):

        super(SelfAttention, self).__init__(**kwargs)

        self.channels = ch

        self.filters_f_g = self.channels // 8

        self.filters_h = self.channels



    def build(self, input_shape):

        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)

        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)



        # Create a trainable weight variable for this layer:

        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)

        self.kernel_f = self.add_weight(shape=kernel_shape_f_g,

                                        initializer='glorot_uniform',

                                        name='kernel_f',

                                        trainable=True)

        self.kernel_g = self.add_weight(shape=kernel_shape_f_g,

                                        initializer='glorot_uniform',

                                        name='kernel_g',

                                        trainable=True)

        self.kernel_h = self.add_weight(shape=kernel_shape_h,

                                        initializer='glorot_uniform',

                                        name='kernel_h',

                                        trainable=True)



        super(SelfAttention, self).build(input_shape)

        # Set input spec.

        self.input_spec = InputSpec(ndim=4,

                                    axes={3: input_shape[-1]})

        self.built = True



    def call(self, x):

        def hw_flatten(x):

            return K.reshape(x, shape=[K.shape(x)[0], K.shape(x)[1]*K.shape(x)[2], K.shape(x)[3]])



        f = K.conv2d(x,

                     kernel=self.kernel_f,

                     strides=(1, 1), padding='same')  # [bs, h, w, c']

        g = K.conv2d(x,

                     kernel=self.kernel_g,

                     strides=(1, 1), padding='same')  # [bs, h, w, c']

        h = K.conv2d(x,

                     kernel=self.kernel_h,

                     strides=(1, 1), padding='same')  # [bs, h, w, c]



        s = K.batch_dot(hw_flatten(g), K.permute_dimensions(hw_flatten(f), (0, 2, 1)))  # # [bs, N, N]



        beta = K.softmax(s, axis=-1)  # attention map



        o = K.batch_dot(beta, hw_flatten(h))  # [bs, N, C]



        o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]

        x = self.gamma * o + x



        return x



    def compute_output_shape(self, input_shape):

        return input_shape
    
    
#def _l2normalize(v, eps=1e-12):
#    return v / tf.sqrt(tf.reduce_sum(tf.square(v)) + eps)
#
#
#def max_singular_value(W, u, Ip=1):
#    _u = u
#    _v = 0
#    for _ in range(Ip):
#        _v = _l2normalize(tf.matmul(_u, W), eps=1e-12)
#        _u = _l2normalize(tf.matmul(_v, W, transpose_b=True), eps=1e-12)
#    _v = tf.stop_gradient(_v)
#    _u = tf.stop_gradient(_u)
#    sigma = tf.reduce_sum(tf.matmul(_u, W) * _v)
#    return sigma, _u, _v
#
#def spectral_normalization(name, W, Ip=1):
#    u = tf.get_variable(name + "_u", [1, W.shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)  # 1 x ch
#    W_mat = tf.transpose(tf.reshape(W, [-1, W.shape[-1]]))
#    sigma, _u, _ = max_singular_value(W_mat, u, Ip)
#    with tf.control_dependencies([tf.assign(u, _u)]):
#        W_sn = W / sigma
#    return W_sn