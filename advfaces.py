import sys
import time
import imp
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from functools import partial
from utils import tfutils


class AdvFaces:
    def __init__(self):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(
            gpu_options=gpu_options,
            allow_soft_placement=True,
            log_device_placement=False,
        )
        self.sess = tf.Session(graph=self.graph, config=tf_config)

    def initialize(self, config, num_classes):
        """
            Initialize the graph from scratch according config.
        """
        with self.graph.as_default():
            with self.sess.as_default():
                G_grad_splits = []
                D_grad_splits = []
                average_dict = {}
                concat_dict = {}

                def insert_dict(_dict, k, v):
                    if k in _dict:
                        _dict[k].append(v)
                    else:
                        _dict[k] = [v]

                # Set up placeholders
                h, w = config.image_size
                channels = config.channels
                self.disc_counter = config.disc_counter

                self.mode = config.mode

                self.aux_matcher = imp.load_source("network_model",
                    config.aux_matcher_definition)

                summaries = []

                self.images = tf.placeholder(
                    tf.float32, shape=[None, h, w, channels], name="images"
                )
                self.t = tf.placeholder(tf.float32, shape=[None, h, w, channels])
                self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
                self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
                self.phase_train = tf.placeholder(tf.bool, name="phase_train")
                self.global_step = tf.Variable(
                    0, trainable=False, dtype=tf.int32, name="global_step"
                )

                self.setup_network_model(config, num_classes)

                if self.mode == "target":
                    self.perturb, self.G = self.generator(self.images, self.t)
                else:
                    self.perturb, self.G = self.generator(self.images)


                ########################## GAN LOSS ###########################
                self.D_real = self.discriminator(self.images)
                self.D_fake = self.discriminator(self.G)
                d_loss_real = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.D_real, labels=tf.ones_like(self.D_real)
                    )
                )
                d_loss_fake = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.D_fake, labels=tf.zeros_like(self.D_fake)
                    )
                )
                g_adv_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.D_fake, labels=tf.ones_like(self.D_fake)
                    )
                )
                self.d_loss = d_loss_real + d_loss_fake

                ########################## IDENTITY LOSS #######################
                with slim.arg_scope(inception_arg_scope()):
                        self.fake_feat, _ = self.aux_matcher.inference(
                            self.G,
                            bottleneck_layer_size=512,
                            phase_train=False,
                            keep_probability=1.0,
                        )
                        if self.mode == "target":
                            self.real_feat, _ = self.aux_matcher.inference(
                                self.t,
                                bottleneck_layer_size=512,
                                phase_train=False,
                                keep_probability=1.0,
                                reuse=True,
                            )
                        else:
                            self.real_feat, _ = self.aux_matcher.inference(
                                self.images,
                                bottleneck_layer_size=512,
                                phase_train=False,
                                keep_probability=1.0,
                                reuse=True,
                            )
                if self.mode == "target":
                    identity_loss = tf.reduce_mean(
                        1.0 - (tfutils.cosine_pair(self.fake_feat, self.real_feat) + 1.0)/ 2.0
                    )
                else:
                    identity_loss = tf.reduce_mean(
                        tfutils.cosine_pair(self.fake_feat, self.real_feat)
                    )
                identity_loss = config.idt_loss_factor * identity_loss

                ########################## PERTURBATION LOSS #####################
                perturb_loss = config.perturb_loss_factor * \
                            tf.reduce_mean(
                                tf.maximum(tf.zeros((tf.shape(self.perturb)[0])) + config.MAX_PERTURBATION,
                                tf.norm(tf.reshape( self.perturb, (tf.shape(self.perturb)[0], -1)),
                            axis=1)))

                ########################## PIXEL LOSS ############################
                pixel_loss = 1000.0 * tf.reduce_mean(tf.abs(self.G - self.images))

                self.g_loss = g_adv_loss + identity_loss + perturb_loss

                ################### LOSS SUMMARY ###################
                insert_dict(average_dict, "g_loss", self.g_loss)
                insert_dict(average_dict, "d_loss", self.d_loss)
                insert_dict(average_dict, "gadv_loss", g_adv_loss)
                insert_dict(average_dict, "idt_loss", identity_loss)
                insert_dict(average_dict, "prt_loss", perturb_loss)
                insert_dict(average_dict, "pxl_loss", pixel_loss)

                ################# VARIABLES TO UPDATE #################
                G_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator"
                )
                D_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator"
                )
                self.train_G_op = tf.train.AdamOptimizer(
                    self.learning_rate, beta1=0.5, beta2=0.9
                ).minimize(self.g_loss, var_list=G_vars)
                self.train_D_op = tf.train.AdamOptimizer(
                    self.learning_rate, beta1=0.5, beta2=0.9
                ).minimize(self.d_loss, var_list=D_vars)

                for k, v in average_dict.items():
                    v = tfutils.average_tensors(v)
                    average_dict[k] = v
                    tfutils.insert(k, v)
                    if "loss" in k:
                        summaries.append(tf.summary.scalar("losses/" + k, v))
                    elif "acc" in k:
                        summaries.append(tf.summary.scalar("acc/" + k, v))
                    else:
                        tf.summary(k, v)
                for k, v in concat_dict.items():
                    v = tf.concat(v, axis=0, name="merged_" + k)
                    concat_dict[k] = v
                    tfutils.insert(k, v)
                trainable_variables = [t for t in tf.trainable_variables()]

                fn = [var for var in tf.trainable_variables() if config.aux_matcher_scope in var.name]
                print(trainable_variables)

                self.update_global_step_op = tf.assign_add(self.global_step, 1)
                summaries.append(tf.summary.scalar("learning_rate", self.learning_rate))
                self.summary_op = tf.summary.merge(summaries)

                self.sess.run(tf.local_variables_initializer())
                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver(trainable_variables, max_to_keep=None)
                f_saver = tf.train.Saver(fn)
                f_saver.restore(
                    self.sess, config.aux_matcher_path
                )

                self.watch_list = tfutils.get_watchlist()

    def setup_network_model(self, config, num_classes):
        network_models = imp.load_source("network_model", config.network)
        self.generator = partial(
            network_models.generator,
            keep_prob=self.keep_prob,
            phase_train=self.phase_train,
            weight_decay=config.weight_decay,
            reuse=tf.AUTO_REUSE,
            scope="Generator",
        )

        self.generator_mask = partial(
            network_models.generator,
            keep_prob=self.keep_prob,
            phase_train=self.phase_train,
            weight_decay=config.weight_decay,
            reuse=tf.AUTO_REUSE,
            scope="Generator",
        )

        self.discriminator = partial(
            network_models.normal_discriminator,
            keep_prob=self.keep_prob,
            phase_train=self.phase_train,
            weight_decay=config.weight_decay,
            reuse=tf.AUTO_REUSE,
            scope="Discriminator",
        )
        

    def train(
        self,
        image_batch,
        target_batch,
        label_batch,
        learning_rate,
        num_classes,
        keep_prob,
    ):
        h, w, c = image_batch.shape[1:]
        feed_dict = {
            self.images: image_batch,
            self.learning_rate: learning_rate,
            self.keep_prob: keep_prob,
            self.t: target_batch,
            self.phase_train: True,
        }
        for i in range(1):
            _ = self.sess.run(self.train_G_op, feed_dict=feed_dict)

        _, wl, sm, step = self.sess.run(
            [
                self.train_D_op,
                tfutils.get_watchlist(),
                self.summary_op,
                self.update_global_step_op,
            ],
            feed_dict=feed_dict,
        )
        return wl, sm, step

    def restore_model(self, *args, **kwargs):
        trainable_variables = self.graph.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES
        )
        tfutils.restore_model(self.sess, trainable_variables, *args, **kwargs)

    def save_model(self, model_dir, global_step):
        tfutils.save_model(self.sess, self.saver, model_dir, global_step)

    def decode_images(self, features, batch_size):
        num_images = features.shape[0]
        h, w, c = tuple(self.G.shape[1:])
        result = np.ndarray((num_images, h, w, c), dtype=np.float32)
        for start_idx in range(0, num_images, batch_size):
            end_idx = min(num_images, start_idx + batch_size)
            feat = features[start_idx:end_idx]
            feed_dict = {
                self.feats: feat,
                self.phase_train: False,
                self.keep_prob: 1.0,
            }
            result[start_idx:end_idx] = self.sess.run(self.G, feed_dict=feed_dict)
        return result

    def generate_images(
        self, images, targets=None, batch_size=128, return_targets=False
    ):
        num_images = images.shape[0]
        h, w, c = tuple(self.G.shape[1:])
        result = np.ndarray((num_images, h, w, c), dtype=np.float32)
        perturb = np.ndarray((num_images, h, w, c), dtype=np.float32)
        for start_idx in range(0, num_images, batch_size):
            end_idx = min(num_images, start_idx + batch_size)
            im = images[start_idx:end_idx]

            if self.mode == "target":
                t = targets[start_idx:end_idx]
                feed_dict = {
                    self.images: im,
                    self.t: t,
                    self.phase_train: False,
                    self.keep_prob: 1.0,
                }
            else:
                feed_dict = {
                    self.images: im,
                    self.phase_train: False,
                    self.keep_prob: 1.0,
                }
            result[start_idx:end_idx], perturb[start_idx:end_idx] = self.sess.run(
                [self.G, self.perturb], feed_dict=feed_dict
            )
        return result, perturb

    def aux_matcher_extract_feature(self, images, batch_size=512, verbose=True):
        num_images = images.shape[0]
        fake = np.ndarray((num_images, 512), dtype=np.float32)
        start_time = time.time()
        for start_idx in range(0, num_images, batch_size):
            if verbose:
                elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))
                sys.stdout.write('# of images: %d Current image: %d Elapsed time: %s \t\r' 
                    % (num_images, start_idx, elapsed_time))
            end_idx = min(num_images, start_idx + batch_size)
            im = images[start_idx:end_idx]
            if self.mode == 'target':
                feed_dict = {
                    self.t: im,
                    self.phase_train: False,
                    self.keep_prob: 1.0,
                }
            else:
                feed_dict = {
                        self.images: im,
                        self.phase_train: False,
                        self.keep_prob: 1.0,
                }
            fake[start_idx:end_idx] = self.sess.run(self.real_feat, feed_dict=feed_dict)
        return fake

    def load_model(self, *args, **kwargs):
        print("load_model")
        tfutils.load_model(self.sess, *args, **kwargs)
        self.phase_train = self.graph.get_tensor_by_name("phase_train:0")
        self.keep_prob = self.graph.get_tensor_by_name("keep_prob:0")
        self.perturb = self.graph.get_tensor_by_name("Generator_1/output:0")
        self.G = self.graph.get_tensor_by_name("Generator_1/sub:0")
        self.D = self.graph.get_tensor_by_name("Discriminator/Reshape:0")
        self.images = self.graph.get_tensor_by_name("images:0")
        self.mode = "obfuscation"
        if self.mode == "target":
            self.t = self.graph.get_tensor_by_name("Placeholder:0")

###############################################################################
####################  ONLY NEEDED FOR FACENET MATCHER #########################
def inception_arg_scope(
    weight_decay=0.00004,
    use_batch_norm=True,
    batch_norm_decay=0.9997,
    batch_norm_epsilon=0.001,
):

    """Defines the default arg scope for inception models.

  Args:

    weight_decay: The weight decay to use for regularizing the model.

    use_batch_norm: "If `True`, batch_norm is applied after each convolution.

    batch_norm_decay: Decay for batch norm moving average.

    batch_norm_epsilon: Small float added to variance to avoid dividing by zero

      in batch norm.

  Returns:

    An `arg_scope` to use for the inception models.

  """

    batch_norm_params = {
        # Decay for the moving averages.
        "decay": batch_norm_decay,
        # epsilon to prevent 0s in variance.
        "epsilon": batch_norm_epsilon,
        # collection containing update_ops.
        "updates_collections": tf.GraphKeys.UPDATE_OPS,
    }

    if use_batch_norm:

        normalizer_fn = slim.batch_norm

        normalizer_params = batch_norm_params

    else:

        normalizer_fn = None

        normalizer_params = {}

    # Set weight_decay for weights in Conv and FC layers.

    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=slim.l2_regularizer(weight_decay),
    ):

        with slim.arg_scope(
            [slim.conv2d],
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params,
        ) as sc:

            return sc
