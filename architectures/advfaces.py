from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import layer_norm, instance_norm

batch_norm_params = {
    "decay": 0.995,
    "epsilon": 0.001,
    "updates_collections": None,
    "variables_collections": [tf.GraphKeys.TRAINABLE_VARIABLES],
}

gf_dim = 64


def leaky_relu(x):
    return tf.maximum(0.2 * x, x)


def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1:
        return x
    with tf.variable_scope("Upscale2d"):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
        x = tf.tile(x, [1, 1, factor, 1, factor, 1])
        x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
        return x


def padding(x, pad, pad_type="reflect"):
    if pad_type == "zero":
        return tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
    if pad_type == "reflect":
        return tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0,]], mode="REFLECT")
    else:
        raise ValueError("Unknown pad type: {}".format(pad_type))


def conv(x, *args, pad=1, **kwargs):
    with slim.arg_scope([slim.conv2d], padding="VALID"):
        x = padding(x, pad)
        return slim.conv2d(x, *args, **kwargs)


def deconv(x, *args, pad=1, **kwargs):
    with slim.arg_scope([slim.conv2d_transpose], padding="SAME"):
        x = padding(x, pad)
        return slim.conv2d(x, *args, **kwargs)


def generator(
    images,
    targets=None,
    keep_prob=1.0,
    phase_train=True,
    weight_decay=0.0,
    reuse=None,
    scope="Generator",
):
    with slim.arg_scope(
        [slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
        activation_fn=tf.nn.relu,
        normalizer_fn=instance_norm,
        normalizer_params=None,
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
        weights_regularizer=slim.l2_regularizer(weight_decay),
    ):
        with tf.variable_scope(scope, [images], reuse=reuse):
            with slim.arg_scope(
                [slim.dropout, slim.batch_norm], is_training=phase_train
            ):
                if targets is not None:
                    net = tf.concat([images, targets], axis=-1)
                else:
                    net = images
                print(
                    "{} input shape: ".format(scope), [dim.value for dim in net.shape],
                )
                k = 64
                net = conv(net, k, kernel_size=7, stride=1, pad=3, scope="conv0")
                print("conv0 shape: ", [dim.value for dim in net.shape])
                net = conv(net, 2 * k, kernel_size=4, stride=2, scope="conv1")
                print("conv1 shape: ", [dim.value for dim in net.shape])
                net = conv(net, 4 * k, kernel_size=4, stride=2, scope="conv2")
                print("conv2 shape: ", [dim.value for dim in net.shape])

                for i in range(3):
                    net_ = conv(net, 4 * k, kernel_size=3, scope="res{}_0".format(i))
                    net += conv(
                        net_,
                        4 * k,
                        3,
                        activation_fn=None,
                        biases_initializer=None,
                        scope="res{}_1".format(i),
                    )
                    print(
                        "res{} shape:".format(i), [dim.value for dim in net.shape],
                    )
                encoded = tf.identity(net, name="encoded")

    with slim.arg_scope(
        [slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
        activation_fn=tf.nn.relu,
        normalizer_fn=layer_norm,
        normalizer_params=None,
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
        weights_regularizer=slim.l2_regularizer(weight_decay),
    ):
        with tf.variable_scope(scope, [encoded], reuse=reuse):
            with slim.arg_scope(
                [slim.dropout, slim.batch_norm], is_training=phase_train
            ):
                with slim.arg_scope(
                    [slim.fully_connected],
                    normalizer_fn=layer_norm,
                    normalizer_params=None,
                ):
                    net = upscale2d(encoded, 2)
                    net = conv(net, 2 * k, 5, pad=2, scope="deconv1_1")
                    print("deconv1 shape:", [dim.value for dim in net.shape])

                    net = upscale2d(net, 2)
                    net = conv(net, k, 5, pad=2, scope="deconv2_1")
                    print("deconv2 shape:", [dim.value for dim in net.shape])

                    net = conv(
                        net,
                        3,
                        7,
                        pad=3,
                        activation_fn=None,
                        normalizer_fn=None,
                        scope="conv_img",
                    )
                    net = tf.nn.tanh(net, name="output")
                    print("output:", [dim.value for dim in net.shape])

                    perturb = tf.clip_by_value(net, -1.0, 1.0)
                    output = (
                        2 * tf.clip_by_value(perturb + (images + 1.0) / 2.0, 0, 1) - 1
                    )
                    return net, output


def normal_discriminator(
    images,
    keep_prob=1.0,
    phase_train=True,
    weight_decay=0.0,
    reuse=None,
    scope="Discriminator",
):
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        activation_fn=leaky_relu,
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params,
    ):
        with tf.variable_scope(scope, [images], reuse=reuse):
            with slim.arg_scope(
                [slim.batch_norm, slim.dropout], is_training=phase_train
            ):

                print(
                    "{} input shape:".format(scope),
                    [dim.value for dim in images.shape],
                )

                net = conv(
                    images,
                    32,
                    kernel_size=4,
                    stride=2,
                    scope="conv1",
                    activation_fn=None,
                )
                print("module_1 shape:", [dim.value for dim in net.shape])

                net = conv(
                    net, 64, kernel_size=4, stride=2, scope="conv2", activation_fn=None,
                )
                print("module_2 shape:", [dim.value for dim in net.shape])

                net = conv(net, 128, kernel_size=4, stride=2, scope="conv3")
                print("module_3 shape:", [dim.value for dim in net.shape])

                net = conv(net, 256, kernel_size=4, stride=2, scope="conv4")
                print("module_4 shape:", [dim.value for dim in net.shape])

                net = conv(net, 512, kernel_size=4, stride=2, scope="conv5")
                print("module_5 shape:", [dim.value for dim in net.shape])

                net = slim.conv2d(
                    net,
                    1,
                    1,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope="patch_logits",
                )
                print("patch:", [dim.value for dim in net.shape])
                net = tf.reshape(net, [-1, 1])
                print("disc shape: ", [dim.value for dim in net.shape])
                return net
