
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import Constraint
import tensorflow as tf


class ZeroThresholdConstraint(Constraint):
    def __init__(self, threshold=1e-4):
        self.threshold = threshold
    def __call__(self, w):
        w = tf.where(tf.abs(w) < self.threshold, tf.zeros_like(w), w)
        return w
    def get_config(self):
        return {'threshold': self.threshold}


def build_fmri_net(temporal_filters=8, num_classes=6,
                   input_shape=(214, 277, 1), depth_multiplier=4,
                   zero_thresh=1e-4, dropout_in=0.25, dropout_mid=0.5, name="fmriNet"):
    
    
    inputs  = Input(shape=input_shape)
    x       = Dropout(dropout_in)(inputs)
    x       = Conv2D(temporal_filters, (1, 60), padding='same', use_bias=False)(x)
    x       = Permute((2, 1, 3))(x)
    x       = DepthwiseConv2D((1, input_shape[0]), use_bias=False,
                              depth_multiplier=depth_multiplier,
                              depthwise_constraint=ZeroThresholdConstraint(threshold=zero_thresh))(x)
    x       = BatchNormalization()(x)
    x       = Activation('relu')(x)
    x       = Permute((2, 1, 3))(x)
    x       = AveragePooling2D((1, 15))(x)
    x       = Dropout(dropout_mid)(x)
    x       = SeparableConv2D(64, (1, 8), padding='same', use_bias=False)(x)
    x       = BatchNormalization()(x)
    x       = Activation('relu')(x)
    x       = AveragePooling2D((1, 4))(x)
    x       = Dropout(dropout_mid)(x)
    
    
    features= Flatten(name='flatten')(x)
    logits  = Dense(num_classes, name='dense')(features)
    softmax = Activation('softmax', name='softmax')(logits)
    
    return Model(inputs=inputs, outputs=softmax, name=name)


def fmriNet8(num_classes=6, input_shape=(214, 277, 1), **kwargs):
    return build_fmri_net(temporal_filters=8, num_classes=num_classes,
                          input_shape=input_shape, name="fmriNet8", **kwargs)

def fmriNet16(num_classes=6, input_shape=(214, 277, 1), **kwargs):
    return build_fmri_net(temporal_filters=16, num_classes=num_classes,
                          input_shape=input_shape, name="fmriNet16", **kwargs)

def fmriNet32(num_classes=6, input_shape=(214, 277, 1), **kwargs):
    return build_fmri_net(temporal_filters=32, num_classes=num_classes,
                          input_shape=input_shape, name="fmriNet32", **kwargs)
